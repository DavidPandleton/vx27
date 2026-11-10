#include "vx27_gpu_persist.h"
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
static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want, int device_local_ok) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
        if ((fl & want) != want) continue;
        if (!device_local_ok && (fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
        return i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & want) == want)) return i;
    return 0;
}

struct vx27_gpu_ctx {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue queue; uint32_t qi;
    VkBuffer wbuf, qbuf, ybuf; VkDeviceMemory wmem, qmem, ymem;
    VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSet ds; VkDescriptorPool dp;
    VkCommandPool pool; VkCommandBuffer cb; VkFence fence;
    int rows, nblocks, B;
    size_t row_bytes;
};

int vx27_gpu_ctx_init(vx27_gpu_ctx **out, const void *Wq4, int rows, int cols, char *err, size_t errlen) {
    if (cols % 256 || rows <= 0) return -2;
    int B = 8, nblocks = cols / 256;
    size_t row_bytes = (size_t)nblocks * sizeof(block_q4_k);
    size_t wsz = (size_t)rows * row_bytes, qsz = (size_t)nblocks * B * sizeof(block_q8_k), ysz = (size_t)rows * B * 4;
    vx27_gpu_ctx *c = calloc(1, sizeof(*c));
    c->rows = rows; c->nblocks = nblocks; c->B = B; c->row_bytes = row_bytes;

    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, NULL, &c->inst) != VK_SUCCESS) { free(c); return -1; }
    uint32_t nd = 1; vkEnumeratePhysicalDevices(c->inst, &nd, &c->pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    free(qf); c->qi = qi;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(c->pd, &dci, NULL, &c->dev) != VK_SUCCESS) { free(c); return -1; }
    vkGetDeviceQueue(c->dev, qi, 0, &c->queue);

    // weight: staging host-visible -> device-local
    VkBuffer sbuf; VkDeviceMemory smem;
    VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = wsz; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(c->dev, &bi, NULL, &sbuf);
    VkMemoryRequirements r1; vkGetBufferMemoryRequirements(c->dev, sbuf, &r1);
    VkMemoryAllocateInfo a1 = {0}; a1.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    a1.allocationSize = r1.size;
    a1.memoryTypeIndex = mem_type(c->pd, r1.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    vkAllocateMemory(c->dev, &a1, NULL, &smem);
    vkBindBufferMemory(c->dev, sbuf, smem, 0);
    void *pm = NULL; vkMapMemory(c->dev, smem, 0, wsz, 0, &pm); memcpy(pm, Wq4, wsz); vkUnmapMemory(c->dev, smem);

    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkCreateBuffer(c->dev, &bi, NULL, &c->wbuf);
    VkMemoryRequirements r2; vkGetBufferMemoryRequirements(c->dev, c->wbuf, &r2);
    VkMemoryAllocateInfo a2 = {0}; a2.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    a2.allocationSize = r2.size;
    a2.memoryTypeIndex = mem_type(c->pd, r2.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1);
    // fallback host-visible if no pure device-local
    vkAllocateMemory(c->dev, &a2, NULL, &c->wmem);
    vkBindBufferMemory(c->dev, c->wbuf, c->wmem, 0);

    // Q/Y host-visible (kecil)
    VkBuffer *bufs[2] = {&c->qbuf, &c->ybuf}; VkDeviceMemory *mems[2] = {&c->qmem, &c->ymem};
    size_t szs[2] = {qsz, ysz};
    for (int i = 0; i < 2; i++) {
        VkBufferCreateInfo b2 = {0}; b2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        b2.size = szs[i]; b2.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; b2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(c->dev, &b2, NULL, bufs[i]);
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(c->dev, *bufs[i], &rq);
        VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        am.allocationSize = rq.size;
        am.memoryTypeIndex = mem_type(c->pd, rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        vkAllocateMemory(c->dev, &am, NULL, mems[i]);
        vkBindBufferMemory(c->dev, *bufs[i], *mems[i], 0);
    }

    // upload staging -> device (one time)
    VkCommandPoolCreateInfo cp0 = {0}; cp0.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cp0.queueFamilyIndex = qi;
    VkCommandPool upool; vkCreateCommandPool(c->dev, &cp0, NULL, &upool);
    VkCommandBufferAllocateInfo ca0 = {0}; ca0.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca0.commandPool = upool; ca0.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca0.commandBufferCount = 1;
    VkCommandBuffer ucb; vkAllocateCommandBuffers(c->dev, &ca0, &ucb);
    VkFenceCreateInfo fi0 = {0}; fi0.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence uf; vkCreateFence(c->dev, &fi0, NULL, &uf);
    VkCommandBufferBeginInfo bb0 = {0}; bb0.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(ucb, &bb0);
    VkBufferCopy cp = {0, 0, wsz};
    vkCmdCopyBuffer(ucb, sbuf, c->wbuf, 1, &cp);
    vkEndCommandBuffer(ucb);
    VkSubmitInfo si0 = {0}; si0.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si0.commandBufferCount = 1; si0.pCommandBuffers = &ucb;
    vkQueueSubmit(c->queue, 1, &si0, uf);
    vkWaitForFences(c->dev, 1, &uf, VK_TRUE, 60000000000ull);

    // pipeline (tiled v1)
    FILE *sf = fopen("build/q4k_matvec_tiled.spv", "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing tiled spv"); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)spv_n);
    fread(code, 1, (size_t)spv_n, sf); fclose(sf);
    VkShaderModuleCreateInfo smi = {0}; smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = (size_t)spv_n; smi.pCode = code;
    VkShaderModule mod; vkCreateShaderModule(c->dev, &smi, NULL, &mod);
    free(code);
    VkDescriptorSetLayoutBinding bind[3] = {0};
    for (int i = 0; i < 3; i++) { bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli = {0}; dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3; dli.pBindings = bind;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(c->dev, &dli, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {0}; pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(c->dev, &pli, NULL, &c->pl);
    VkComputePipelineCreateInfo cpi = {0}; cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main"; cpi.layout = c->pl;
    vkCreateComputePipelines(c->dev, NULL, 1, &cpi, NULL, &c->pipe);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi = {0}; dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    vkCreateDescriptorPool(c->dev, &dpi, NULL, &c->dp);
    VkDescriptorSetAllocateInfo dsa = {0}; dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = c->dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    vkAllocateDescriptorSets(c->dev, &dsa, &c->ds);
    VkDescriptorBufferInfo bbi[3] = {{c->wbuf,0,VK_WHOLE_SIZE},{c->qbuf,0,VK_WHOLE_SIZE},{c->ybuf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3] = {0};
    for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = c->ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bbi[i]; }
    vkUpdateDescriptorSets(c->dev, 3, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {0}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = qi;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(c->dev, &cpci, NULL, &c->pool);
    VkCommandBufferAllocateInfo cba = {0}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = c->pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(c->dev, &cba, &c->cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(c->dev, &fi, NULL, &c->fence);
    *out = c;
    return 0;
}

int vx27_gpu_ctx_dispatch(vx27_gpu_ctx *c, const float *xB, float *y_rowmajor, double *ms, char *err, size_t errlen) {
    int rows = c->rows, nblocks = c->nblocks, B = c->B;
    size_t qsz = (size_t)nblocks * B * sizeof(block_q8_k), ysz = (size_t)rows * B * 4;
    block_q8_k *q8 = malloc(qsz);
    for (int t = 0; t < B; t++) quantize_q8_k(xB + (size_t)t * (nblocks * 256), q8 + (size_t)t * nblocks, nblocks);
    void *pq = NULL, *py = NULL;
    vkMapMemory(c->dev, c->qmem, 0, qsz, 0, &pq); memcpy(pq, q8, qsz); vkUnmapMemory(c->dev, c->qmem);
    free(q8);
    double t0 = now_ms();
    VkCommandBufferBeginInfo cbb = {0}; cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(c->cb, &cbb);
    vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe);
    vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &c->ds, 0, NULL);
    struct { uint32_t B, nblocks, rows; } pc = {(uint32_t)B, (uint32_t)nblocks, (uint32_t)rows};
    vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(c->cb, (uint32_t)rows, 1, 1);
    VkMemoryBarrier mb = {0}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(c->cb);
    vkResetFences(c->dev, 1, &c->fence);
    VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &c->cb;
    vkQueueSubmit(c->queue, 1, &si, c->fence);
    vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, 60000000000ull);
    double t1 = now_ms();
    vkMapMemory(c->dev, c->ymem, 0, ysz, 0, &py); memcpy(y_rowmajor, py, ysz); vkUnmapMemory(c->dev, c->ymem);
    vkResetCommandBuffer(c->cb, 0);
    if (ms) *ms = t1 - t0;
    (void)err; (void)errlen;
    return 0;
}

void vx27_gpu_ctx_free(vx27_gpu_ctx *c) { free(c); }
