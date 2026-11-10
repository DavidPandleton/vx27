#include "vx27_gpu.h"
#include "quant.h"
#include "quant_avx2.h"
#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0;
}

int vx27_gpu_device_count(void) {
    VkApplicationInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    VkInstance inst = NULL;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) return -1;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    vkDestroyInstance(inst, NULL);
    return (int)n;
}

int vx27_gpu_q4_bench(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B,
                      double *gpu_ms, double *cpu_ms, double *gvals,
                      float *y_gpu0, float *y_cpu0, char *err, size_t errlen) {
    if (cols % 256 != 0 || rows <= 0 || B <= 0 || B > 16) {
        if (err) snprintf(err, errlen, "bad dims");
        return -2;
    }
    int nblocks = cols / 256;
    size_t row_bytes = (size_t)nblocks * sizeof(block_q4_k);
    const block_q4_k *W = (const block_q4_k *)Wq4;

    block_q8_k *q8 = (block_q8_k*)malloc((size_t)nblocks * B * sizeof(block_q8_k));
    float *y_cpu = (float*)malloc((size_t)rows * B * 4);
    if (!q8 || !y_cpu) { free(q8); free(y_cpu); return -2; }
    for (int t = 0; t < B; t++)
        quantize_q8_k(xfloat + (size_t)t * cols, q8 + (size_t)t * nblocks, nblocks);

    // CPU baseline: all rows x B tokens via AVX2 single-dot
    double t0 = now_ms();
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < B; t++)
            y_cpu[(size_t)r * B + t] = dot_q4_k_q8_k_avx2(
                (const block_q4_k*)((const uint8_t*)W + (size_t)r * row_bytes),
                q8 + (size_t)t * nblocks, (size_t)nblocks);
    double t1 = now_ms();
    if (cpu_ms) *cpu_ms = t1 - t0;
    if (y_cpu0) *y_cpu0 = y_cpu[0];

    // Vulkan setup (host-visible buffers, same as ql gpu_q4k)
    FILE *sf = fopen("build/q4k_matvec.spv", "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing build/q4k_matvec.spv"); free(q8); free(y_cpu); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = (uint32_t*)malloc((size_t)spv_n);
    if (!code || fread(code, 1, (size_t)spv_n, sf) != (size_t)spv_n) { fclose(sf); free(code); free(q8); free(y_cpu); return -1; }
    fclose(sf);

    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { free(code); free(q8); free(y_cpu); return -1; }
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = (VkQueueFamilyProperties*)malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    if (qi == nq) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
    VkQueue queue; vkGetDeviceQueue(dev, qi, 0, &queue);

    VkBuffer bw, bq, by; VkDeviceMemory mw, mq, my; void *pw = NULL, *pq = NULL, *py = NULL;
    size_t wsz = (size_t)rows * row_bytes, qsz = (size_t)nblocks * B * sizeof(block_q8_k), ysz = (size_t)rows * B * 4;
    VkBuffer *bufs[3] = {&bw, &bq, &by}; VkDeviceMemory *mems[3] = {&mw, &mq, &my};
    size_t szs[3] = {wsz, qsz, ysz}; void **maps[3] = {&pw, &pq, &py};
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = szs[i]; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(dev, &bi, NULL, bufs[i]) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, *bufs[i], &req);
        VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        am.allocationSize = req.size;
        am.memoryTypeIndex = mem_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(dev, &am, NULL, mems[i]) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
        vkBindBufferMemory(dev, *bufs[i], *mems[i], 0);
        vkMapMemory(dev, *mems[i], 0, szs[i], 0, maps[i]);
    }
    memcpy(pw, W, wsz); memcpy(pq, q8, qsz); memset(py, 0, ysz);

    VkShaderModuleCreateInfo smi = {0}; smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = (size_t)spv_n; smi.pCode = code;
    VkShaderModule mod; vkCreateShaderModule(dev, &smi, NULL, &mod);
    VkDescriptorSetLayoutBinding bind[3] = {0};
    for (int i = 0; i < 3; i++) { bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli = {0}; dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3; dli.pBindings = bind;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {0}; pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &pli, NULL, &pl);
    VkComputePipelineCreateInfo cpi = {0}; cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main"; cpi.layout = pl;
    VkPipeline pipe; vkCreateComputePipelines(dev, NULL, 1, &cpi, NULL, &pipe);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi = {0}; dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpi, NULL, &dp);
    VkDescriptorSetAllocateInfo dsa = {0}; dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsa, &ds);
    VkDescriptorBufferInfo bi[3] = {{bw,0,VK_WHOLE_SIZE},{bq,0,VK_WHOLE_SIZE},{by,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3] = {0};
    for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {0}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = qi;
    VkCommandPool pool; vkCreateCommandPool(dev, &cpci, NULL, &pool);
    VkCommandBufferAllocateInfo cba = {0}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &cba, &cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; vkCreateFence(dev, &fi, NULL, &fence);

    uint32_t total = (uint32_t)rows * (uint32_t)B, groups = (total + 63) / 64;
    double g0 = now_ms();
    VkCommandBufferBeginInfo cbb = {0}; cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &cbb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    struct { uint32_t B, nblocks, rows; } pc = {(uint32_t)B, (uint32_t)nblocks, (uint32_t)rows};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, groups, 1, 1);
    VkMemoryBarrier mb = {0}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull);
    double g1 = now_ms();

    float got = ((float*)py)[0];
    if (gpu_ms) *gpu_ms = g1 - g0;
    if (gvals) *gvals = (double)cols * rows * B / ((g1 - g0) * 1e6);
    if (y_gpu0) *y_gpu0 = got;

    // MATCH check on y[0] like ql (2% tolerance)
    float want = y_cpu[0];
    int match = fabsf(got - want) <= 0.02f * fabsf(want) + 0.02f;

    free(code); free(q8); free(y_cpu); free(qf);
    // Note: Vulkan handles leaked on purpose for bench brevity (process exits); test only.
    if (!match) { if (err) snprintf(err, errlen, "DIFF gpu=%.6f cpu=%.6f", got, want); return 1; }
    return 0;
}
int vx27_gpu_q4_bench_tiled(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B,
                      double *gpu_ms, double *cpu_ms, double *gvals,
                      float *y_gpu0, float *y_cpu0, char *err, size_t errlen) {
    if (cols % 256 != 0 || rows <= 0 || B <= 0 || B > 16) {
        if (err) snprintf(err, errlen, "bad dims");
        return -2;
    }
    int nblocks = cols / 256;
    size_t row_bytes = (size_t)nblocks * sizeof(block_q4_k);
    const block_q4_k *W = (const block_q4_k *)Wq4;

    block_q8_k *q8 = (block_q8_k*)malloc((size_t)nblocks * B * sizeof(block_q8_k));
    float *y_cpu = (float*)malloc((size_t)rows * B * 4);
    if (!q8 || !y_cpu) { free(q8); free(y_cpu); return -2; }
    for (int t = 0; t < B; t++)
        quantize_q8_k(xfloat + (size_t)t * cols, q8 + (size_t)t * nblocks, nblocks);

    // CPU baseline: all rows x B tokens via AVX2 single-dot
    double t0 = now_ms();
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < B; t++)
            y_cpu[(size_t)r * B + t] = dot_q4_k_q8_k_avx2(
                (const block_q4_k*)((const uint8_t*)W + (size_t)r * row_bytes),
                q8 + (size_t)t * nblocks, (size_t)nblocks);
    double t1 = now_ms();
    if (cpu_ms) *cpu_ms = t1 - t0;
    if (y_cpu0) *y_cpu0 = y_cpu[0];

    // Vulkan setup (host-visible buffers, same as ql gpu_q4k)
    FILE *sf = fopen("build/q4k_matvec_tiled.spv", "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing build/q4k_matvec_tiled.spv"); free(q8); free(y_cpu); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = (uint32_t*)malloc((size_t)spv_n);
    if (!code || fread(code, 1, (size_t)spv_n, sf) != (size_t)spv_n) { fclose(sf); free(code); free(q8); free(y_cpu); return -1; }
    fclose(sf);

    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { free(code); free(q8); free(y_cpu); return -1; }
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = (VkQueueFamilyProperties*)malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    if (qi == nq) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
    VkQueue queue; vkGetDeviceQueue(dev, qi, 0, &queue);

    VkBuffer bw, bq, by; VkDeviceMemory mw, mq, my; void *pw = NULL, *pq = NULL, *py = NULL;
    size_t wsz = (size_t)rows * row_bytes, qsz = (size_t)nblocks * B * sizeof(block_q8_k), ysz = (size_t)rows * B * 4;
    VkBuffer *bufs[3] = {&bw, &bq, &by}; VkDeviceMemory *mems[3] = {&mw, &mq, &my};
    size_t szs[3] = {wsz, qsz, ysz}; void **maps[3] = {&pw, &pq, &py};
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = szs[i]; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(dev, &bi, NULL, bufs[i]) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, *bufs[i], &req);
        VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        am.allocationSize = req.size;
        am.memoryTypeIndex = mem_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(dev, &am, NULL, mems[i]) != VK_SUCCESS) { free(qf); free(code); free(q8); free(y_cpu); return -1; }
        vkBindBufferMemory(dev, *bufs[i], *mems[i], 0);
        vkMapMemory(dev, *mems[i], 0, szs[i], 0, maps[i]);
    }
    memcpy(pw, W, wsz); memcpy(pq, q8, qsz); memset(py, 0, ysz);

    VkShaderModuleCreateInfo smi = {0}; smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = (size_t)spv_n; smi.pCode = code;
    VkShaderModule mod; vkCreateShaderModule(dev, &smi, NULL, &mod);
    VkDescriptorSetLayoutBinding bind[3] = {0};
    for (int i = 0; i < 3; i++) { bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli = {0}; dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3; dli.pBindings = bind;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {0}; pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &pli, NULL, &pl);
    VkComputePipelineCreateInfo cpi = {0}; cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main"; cpi.layout = pl;
    VkPipeline pipe; vkCreateComputePipelines(dev, NULL, 1, &cpi, NULL, &pipe);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi = {0}; dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpi, NULL, &dp);
    VkDescriptorSetAllocateInfo dsa = {0}; dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsa, &ds);
    VkDescriptorBufferInfo bi[3] = {{bw,0,VK_WHOLE_SIZE},{bq,0,VK_WHOLE_SIZE},{by,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3] = {0};
    for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {0}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = qi;
    VkCommandPool pool; vkCreateCommandPool(dev, &cpci, NULL, &pool);
    VkCommandBufferAllocateInfo cba = {0}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &cba, &cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; vkCreateFence(dev, &fi, NULL, &fence);

    uint32_t groups = (uint32_t)rows;
    double g0 = now_ms();
    VkCommandBufferBeginInfo cbb = {0}; cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &cbb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    struct { uint32_t B, nblocks, rows; } pc = {(uint32_t)B, (uint32_t)nblocks, (uint32_t)rows};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, groups, 1, 1);
    VkMemoryBarrier mb = {0}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull);
    double g1 = now_ms();

    float got = ((float*)py)[0];
    if (gpu_ms) *gpu_ms = g1 - g0;
    if (gvals) *gvals = (double)cols * rows * B / ((g1 - g0) * 1e6);
    if (y_gpu0) *y_gpu0 = got;

    // MATCH check on y[0] like ql (2% tolerance)
    float want = y_cpu[0];
    int match = fabsf(got - want) <= 0.02f * fabsf(want) + 0.02f;

    free(code); free(q8); free(y_cpu); free(qf);
    // Note: Vulkan handles leaked on purpose for bench brevity (process exits); test only.
    if (!match) { if (err) snprintf(err, errlen, "DIFF gpu=%.6f cpu=%.6f", got, want); return 1; }
    return 0;
}
