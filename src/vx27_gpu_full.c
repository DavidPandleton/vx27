#include "vx27_gpu.h"
#include "quant.h"
#include "quant_avx2.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0;
}

// Generic SPV runner: copies full Y (rows*B) out. groups: total workgroups.
static int run_spv(const char *spv, const void *W, const void *Q8,
                   int rows, int nblocks, int B, uint32_t groups, size_t w_row_bytes,
                   float *y_out, double *ms, char *err, size_t errlen) {
    FILE *sf = fopen(spv, "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing %s", spv); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)spv_n);
    if (!code || fread(code, 1, (size_t)spv_n, sf) != (size_t)spv_n) { fclose(sf); free(code); return -1; }
    fclose(sf);
    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { free(code); return -1; }
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    if (qi == nq) { free(qf); free(code); return -1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { free(qf); free(code); return -1; }
    VkQueue queue; vkGetDeviceQueue(dev, qi, 0, &queue);
    size_t row_bytes = w_row_bytes;
    size_t wsz = (size_t)rows * row_bytes, qsz = (size_t)nblocks * B * sizeof(block_q8_k), ysz = (size_t)rows * B * 4;
    VkBuffer bw, bq, by; VkDeviceMemory mw, mq, my; void *pw = NULL, *pq = NULL, *py = NULL;
    VkBuffer *bufs[3] = {&bw, &bq, &by}; VkDeviceMemory *mems[3] = {&mw, &mq, &my};
    size_t szs[3] = {wsz, qsz, ysz}; void **maps[3] = {&pw, &pq, &py};
    for (int i = 0; i < 3; i++) {
        VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = szs[i]; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(dev, &bi, NULL, bufs[i]);
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, *bufs[i], &req);
        VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        am.allocationSize = req.size;
        am.memoryTypeIndex = mem_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &am, NULL, mems[i]);
        vkBindBufferMemory(dev, *bufs[i], *mems[i], 0);
        vkMapMemory(dev, *mems[i], 0, szs[i], 0, maps[i]);
    }
    memcpy(pw, W, wsz); memcpy(pq, Q8, qsz); memset(py, 0, ysz);
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
    double t0 = now_ms();
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
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;
    if (y_out) memcpy(y_out, py, ysz);
    free(code); free(qf);
    return 0;
}

static int full_check(const char *spv, uint32_t groups, const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen) {
    if (cols % 256 != 0 || rows <= 0 || B <= 0 || B > 16) return -2;
    int nblocks = cols / 256;
    block_q8_k *q8 = malloc((size_t)nblocks * B * sizeof(block_q8_k));
    float *ycpu = malloc((size_t)rows * B * 4);
    float *ygpu = malloc((size_t)rows * B * 4);
    if (!q8 || !ycpu || !ygpu) { free(q8); free(ycpu); free(ygpu); return -2; }
    for (int t = 0; t < B; t++)
        quantize_q8_k(xfloat + (size_t)t * cols, q8 + (size_t)t * nblocks, nblocks);
    size_t row_bytes = (size_t)nblocks * sizeof(block_q4_k);
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < B; t++)
            ycpu[(size_t)r * B + t] = dot_q4_k_q8_k_avx2(
                (const block_q4_k*)((const uint8_t*)Wq4 + (size_t)r * row_bytes),
                q8 + (size_t)t * nblocks, (size_t)nblocks);
    double ms = 0;
    int rc = run_spv(spv, Wq4, q8, rows, nblocks, B, groups, row_bytes, ygpu, &ms, err, errlen);
    if (rc != 0) { free(q8); free(ycpu); free(ygpu); return rc; }
    float md = 0; double denom = 0, num = 0;
    for (size_t i = 0; i < (size_t)rows * B; i++) {
        float d = fabsf(ygpu[i] - ycpu[i]);
        if (d > md) md = d;
        denom += (double)ycpu[i] * ycpu[i]; num += (double)d * d;
    }
    double nrmse = sqrt(num / (denom + 1e-12));
    if (gpu_ms) *gpu_ms = ms;
    if (y_rowmajor_out) memcpy(y_rowmajor_out, ygpu, (size_t)rows * B * 4);
    free(q8); free(ycpu); free(ygpu);
    if (!(md <= 0.02f * 8.0f + 0.05f && nrmse < 0.02)) {
        if (err) snprintf(err, errlen, "FULL DIFF max=%.4f nrmse=%.4f", (double)md, nrmse);
        return 1;
    }
    return 0;
}

int vx27_gpu_q4_bench_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen) {
    uint32_t groups = ((uint32_t)rows * (uint32_t)B + 63) / 64;
    return full_check("build/q4k_matvec.spv", groups, Wq4, xfloat, rows, cols, B, y_rowmajor_out, gpu_ms, err, errlen);
}
int vx27_gpu_q4_bench_tiled_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen) {
    uint32_t groups = (uint32_t)rows;
    return full_check("build/q4k_matvec_tiled.spv", groups, Wq4, xfloat, rows, cols, B, y_rowmajor_out, gpu_ms, err, errlen);
}
int vx27_gpu_q4_bench_tiled8_full(const void *Wq4, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen) {
    uint32_t groups = (uint32_t)rows;
    return full_check("build/q4k_matvec_tiled8.spv", groups, Wq4, xfloat, rows, cols, B, y_rowmajor_out, gpu_ms, err, errlen);
}

// Q8_K x Q8_K penuh (format .vx27 Q8, mis. gate_proj Q8 dari forge bits=8).
static int full_check_q8(const char *spv, uint32_t groups, const void *Wq8, const float *xfloat,
                         int rows, int cols, int B, float *y_rowmajor_out,
                         double *gpu_ms, char *err, size_t errlen) {
    if (cols % 256 != 0 || rows <= 0 || B <= 0 || B > 16) return -2;
    int nblocks = cols / 256;
    block_q8_k *q8 = malloc((size_t)nblocks * B * sizeof(block_q8_k));
    float *ycpu = malloc((size_t)rows * B * 4);
    float *ygpu = malloc((size_t)rows * B * 4);
    if (!q8 || !ycpu || !ygpu) { free(q8); free(ycpu); free(ygpu); return -2; }
    for (int t = 0; t < B; t++)
        quantize_q8_k(xfloat + (size_t)t * cols, q8 + (size_t)t * nblocks, nblocks);
    size_t row_bytes = (size_t)nblocks * sizeof(block_q8_k);
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < B; t++) {
            const block_q8_k *W = (const block_q8_k*)((const uint8_t*)Wq8 + (size_t)r * row_bytes);
            const block_q8_k *X = q8 + (size_t)t * nblocks;
            double s = 0;
            for (int b = 0; b < nblocks; b++) {
                double dot = 0;
                for (int i = 0; i < 256; i++) dot += (double)W[b].qs[i] * X[b].qs[i];
                s += dot * (double)W[b].d * X[b].d;
            }
            ycpu[(size_t)r * B + t] = (float)s;
        }
    double ms = 0;
    int rc = run_spv(spv, Wq8, q8, rows, nblocks, B, groups, row_bytes, ygpu, &ms, err, errlen);
    if (rc != 0) { free(q8); free(ycpu); free(ygpu); return rc; }
    float md = 0; double denom = 0, num = 0;
    for (size_t i = 0; i < (size_t)rows * B; i++) {
        float d = fabsf(ygpu[i] - ycpu[i]);
        if (d > md) md = d;
        denom += (double)ycpu[i] * ycpu[i]; num += (double)d * d;
    }
    double nrmse = sqrt(num / (denom + 1e-12));
    if (gpu_ms) *gpu_ms = ms;
    if (y_rowmajor_out) memcpy(y_rowmajor_out, ygpu, (size_t)rows * B * 4);
    free(q8); free(ycpu); free(ygpu);
    if (!(md <= 0.05f && nrmse < 0.02)) {
        if (err) snprintf(err, errlen, "Q8 FULL DIFF max=%.4f nrmse=%.4f", (double)md, nrmse);
        return 1;
    }
    return 0;
}

int vx27_gpu_q8_bench_tiled_full(const void *Wq8, const float *xfloat,
                      int rows, int cols, int B, float *y_rowmajor_out,
                      double *gpu_ms, char *err, size_t errlen) {
    uint32_t groups = (uint32_t)rows;
    return full_check_q8("build/q8_matvec_tiled.spv", groups, Wq8, xfloat, rows, cols, B, y_rowmajor_out, gpu_ms, err, errlen);
}
