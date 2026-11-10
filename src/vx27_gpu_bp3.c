#include "vx27_gpu_bp3.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) { return (double)clock()*1000.0/CLOCKS_PER_SEC; }
#endif

static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & want) == want)) return i;
    return 0;
}

struct vx27_bctx {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue queue; uint32_t qi;
    VkBuffer wbuf, qbuf, ybuf; VkDeviceMemory wmem, qmem, ymem;
    VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSet ds; VkDescriptorPool dp;
    VkCommandPool pool; VkCommandBuffer cb; VkFence fence;
    int rows, cols, Bmax;
};

static int make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags use,
                    VkMemoryPropertyFlags mem, VkBuffer *b, VkDeviceMemory *m, void **map) {
    VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size ? size : 4; bi.usage = use; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, NULL, b) != VK_SUCCESS) return -1;
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, *b, &rq);
    VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    am.allocationSize = rq.size;
    am.memoryTypeIndex = mem_type(pd, rq.memoryTypeBits, mem);
    if (vkAllocateMemory(dev, &am, NULL, m) != VK_SUCCESS) return -1;
    vkBindBufferMemory(dev, *b, *m, 0);
    if (map) { if (vkMapMemory(dev, *m, 0, size ? size : 4, 0, map) != VK_SUCCESS) return -1; }
    return 0;
}

int vx27_bctx_init(vx27_bctx **out, const void *Wbp3, int rows, int cols, char *err, size_t errlen) {
    if (cols % 128 || rows <= 0) { if (err) snprintf(err, errlen, "dims hrs kelipatan 128"); return -2; }
    int Bmax = 8;
    size_t ng = (size_t)cols / 128;
    size_t wsz = (size_t)rows * ng * 13 * 4; // 13 u32 per grup
    size_t qsz = (size_t)Bmax * cols * 4, ysz = (size_t)rows * Bmax * 4;
    vx27_bctx *c = calloc(1, sizeof(*c));
    if (!c) return -1;
    c->rows = rows; c->cols = cols; c->Bmax = Bmax;
    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, NULL, &c->inst) != VK_SUCCESS) { free(c); return -1; }
    uint32_t nd = 1; vkEnumeratePhysicalDevices(c->inst, &nd, &c->pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    free(qf); c->qi = qi; if (qi == nq) { free(c); return -1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(c->pd, &dci, NULL, &c->dev) != VK_SUCCESS) { free(c); return -1; }
    vkGetDeviceQueue(c->dev, qi, 0, &c->queue);

    // staging upload bobot -> DEVICE_LOCAL (sekali)
    VkBuffer sbuf; VkDeviceMemory smem; void *pm = NULL;
    if (make_buf(c->pd, c->dev, wsz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &sbuf, &smem, &pm)) { free(c); return -1; }
    memcpy(pm, Wbp3, wsz); vkUnmapMemory(c->dev, smem);
    if (make_buf(c->pd, c->dev, wsz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &c->wbuf, &c->wmem, NULL)) { free(c); return -1; }
    if (make_buf(c->pd, c->dev, qsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &c->qbuf, &c->qmem, NULL)) { free(c); return -1; }
    if (make_buf(c->pd, c->dev, ysz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &c->ybuf, &c->ymem, NULL)) { free(c); return -1; }
    // copy staging->device
    VkCommandPoolCreateInfo cpi = {0}; cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpi.queueFamilyIndex = qi;
    VkCommandPool up; vkCreateCommandPool(c->dev, &cpi, NULL, &up);
    VkCommandBufferAllocateInfo cai = {0}; cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = up; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer ucb; vkAllocateCommandBuffers(c->dev, &cai, &ucb);
    VkFenceCreateInfo ffi = {0}; ffi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence uf; vkCreateFence(c->dev, &ffi, NULL, &uf);
    VkCommandBufferBeginInfo bb = {0}; bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(ucb, &bb);
    VkBufferCopy cp = {0, 0, wsz};
    vkCmdCopyBuffer(ucb, sbuf, c->wbuf, 1, &cp);
    vkEndCommandBuffer(ucb);
    VkSubmitInfo si0 = {0}; si0.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si0.commandBufferCount = 1; si0.pCommandBuffers = &ucb;
    vkQueueSubmit(c->queue, 1, &si0, uf);
    vkWaitForFences(c->dev, 1, &uf, VK_TRUE, 60000000000ull);

    // pipeline tiled bp3
    FILE *sf = fopen("build/bp3_matvec_tiled.spv", "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing spv"); free(c); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)spv_n);
    fread(code, 1, (size_t)spv_n, sf); fclose(sf);
    VkShaderModuleCreateInfo smi = {0}; smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = (size_t)spv_n; smi.pCode = code;
    VkShaderModule mod; vkCreateShaderModule(c->dev, &smi, NULL, &mod); free(code);
    VkDescriptorSetLayoutBinding bnd[3] = {0};
    for (int i = 0; i < 3; i++) { bnd[i].binding = i; bnd[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[i].descriptorCount = 1; bnd[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli = {0}; dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3; dli.pBindings = bnd;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(c->dev, &dli, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {0}; pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(c->dev, &pli, NULL, &c->pl);
    VkComputePipelineCreateInfo cci = {0}; cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cci.stage.module = mod; cci.stage.pName = "main"; cci.layout = c->pl;
    vkCreateComputePipelines(c->dev, NULL, 1, &cci, NULL, &c->pipe);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi = {0}; dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    vkCreateDescriptorPool(c->dev, &dpi, NULL, &c->dp);
    VkDescriptorSetAllocateInfo dsa = {0}; dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = c->dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    vkAllocateDescriptorSets(c->dev, &dsa, &c->ds);
    VkDescriptorBufferInfo bi[3] = {{c->wbuf,0,VK_WHOLE_SIZE},{c->qbuf,0,VK_WHOLE_SIZE},{c->ybuf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3] = {0};
    for (int i = 0; i < 3; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = c->ds; wr[i].dstBinding = i; wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(c->dev, 3, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {0}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = qi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(c->dev, &cpci, NULL, &c->pool);
    VkCommandBufferAllocateInfo cba = {0}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = c->pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(c->dev, &cba, &c->cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(c->dev, &fi, NULL, &c->fence);
    *out = c;
    return 0;
}

int vx27_bctx_dispatch(vx27_bctx *c, const float *xB, int B, float *y_rowmajor, double *ms, char *err, size_t errlen) {
    if (B < 1 || B > c->Bmax) { if (err) snprintf(err, errlen, "B range"); return -2; }
    size_t qsz = (size_t)B * c->cols * 4, ysz = (size_t)c->rows * B * 4;
    void *pq = NULL, *py = NULL;
    vkMapMemory(c->dev, c->qmem, 0, qsz, 0, &pq); memcpy(pq, xB, qsz); vkUnmapMemory(c->dev, c->qmem);
    double t0 = now_ms();
    VkCommandBufferBeginInfo cbb = {0}; cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(c->cb, &cbb);
    vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe);
    vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &c->ds, 0, NULL);
    struct { uint32_t B, ncols, rows; } pc = {(uint32_t)B, (uint32_t)c->cols, (uint32_t)c->rows};
    vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(c->cb, (uint32_t)c->rows, 1, 1);
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
    return 0;
}

void vx27_bctx_free(vx27_bctx *c) { free(c); }
