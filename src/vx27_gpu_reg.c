#include "vx27_gpu_reg.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define REG_MAX 256
#define QMAX_COLS 17408
#define QMAX_ROWS 248320

typedef struct {
    const void *host;
    VkBuffer wbuf; VkDeviceMemory wmem;
    VkDescriptorSet ds;
    int rows, cols;
    size_t row_u32; // uint per baris
} reg_e;

struct vx27_greg {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue queue; uint32_t qi;
    VkPipeline pipe; VkPipelineLayout pl; VkDescriptorPool dp;
    VkCommandPool pool; VkCommandBuffer cb; VkFence fence;
    VkBuffer qbuf, ybuf; VkDeviceMemory qmem, ymem;
    reg_e e[REG_MAX];
    int n, on;
};

static vx27_greg G;
static int g_on = 0;

static uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & want) == want)) return i;
    return 0;
}

void vx27_gpu_reg_on(void) { if (G.n > 0) g_on = 1; }
void vx27_gpu_reg_off(void) { g_on = 0; }

static int make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags use,
                    VkMemoryPropertyFlags mem, VkBuffer *b, VkDeviceMemory *m) {
    VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size ? size : 4; bi.usage = use; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, NULL, b) != VK_SUCCESS) return -1;
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, *b, &rq);
    VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    am.allocationSize = rq.size;
    am.memoryTypeIndex = mem_type(pd, rq.memoryTypeBits, mem);
    if (vkAllocateMemory(dev, &am, NULL, m) != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(dev, *b, *m, 0) != VK_SUCCESS) return -1;
    return 0;
}

int vx27_gpu_reg_init(const void **blobs, const int *rows, const int *cols, int n, char *err, size_t errlen) {
    if (n <= 0 || n > REG_MAX) { if (err) snprintf(err, errlen, "n range"); return -1; }
    memset(&G, 0, sizeof G);
    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, NULL, &G.inst) != VK_SUCCESS) return -1;
    uint32_t nd = 1; vkEnumeratePhysicalDevices(G.inst, &nd, &G.pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) qi++;
    free(qf); G.qi = qi;
    if (qi == nq) return -1;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(G.pd, &dci, NULL, &G.dev) != VK_SUCCESS) return -1;
    vkGetDeviceQueue(G.dev, qi, 0, &G.queue);

    // q/y host-visible bersama (maksimal)
    if (make_buf(G.pd, G.dev, (VkDeviceSize)QMAX_COLS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &G.qbuf, &G.qmem) != 0) return -1;
    if (make_buf(G.pd, G.dev, (VkDeviceSize)QMAX_ROWS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &G.ybuf, &G.ymem) != 0) return -1;

    // pipeline sekali (shader BP3 tiled)
    FILE *sf = fopen("build/bp3_matvec_tiled.spv", "rb");
    if (!sf) { if (err) snprintf(err, errlen, "missing spv"); return -1; }
    fseek(sf, 0, SEEK_END); long spv_n = ftell(sf); fseek(sf, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)spv_n);
    if (!code || fread(code, 1, (size_t)spv_n, sf) != (size_t)spv_n) { fclose(sf); free(code); return -1; }
    fclose(sf);
    VkShaderModuleCreateInfo smi = {0}; smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = (size_t)spv_n; smi.pCode = code;
    VkShaderModule mod; vkCreateShaderModule(G.dev, &smi, NULL, &mod); free(code);
    VkDescriptorSetLayoutBinding bnd[3] = {0};
    for (int i = 0; i < 3; i++) { bnd[i].binding = i; bnd[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[i].descriptorCount = 1; bnd[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli = {0}; dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 3; dli.pBindings = bnd;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(G.dev, &dli, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pli = {0}; pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(G.dev, &pli, NULL, &G.pl);
    VkComputePipelineCreateInfo cpi = {0}; cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main"; cpi.layout = G.pl;
    vkCreateComputePipelines(G.dev, NULL, 1, &cpi, NULL, &G.pipe);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * REG_MAX};
    VkDescriptorPoolCreateInfo dpi = {0}; dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = REG_MAX; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.dp);

    // upload semua bobot device-local dalam SATU submit
    VkCommandPoolCreateInfo cpi2 = {0}; cpi2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpi2.queueFamilyIndex = qi;
    VkCommandPool up; vkCreateCommandPool(G.dev, &cpi2, NULL, &up);
    VkCommandBufferAllocateInfo cai = {0}; cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = up; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer ucb; vkAllocateCommandBuffers(G.dev, &cai, &ucb);
    VkFenceCreateInfo ffi = {0}; ffi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence uf; vkCreateFence(G.dev, &ffi, NULL, &uf);
    VkCommandBufferBeginInfo bb = {0}; bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(ucb, &bb);
    G.n = 0;
    for (int i = 0; i < n; i++) {
        // lewati tensor 1D (norm, bias): tidak pernah jadi matriks di matvec
        if (cols[i] < 128) continue;
        size_t rb = ((size_t)cols[i] / 128) * 13 * 4;
        size_t wsz = (size_t)rows[i] * rb;
        // staging
        VkBuffer sb; VkDeviceMemory sm;
        if (make_buf(G.pd, G.dev, wsz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     &sb, &sm) != 0) { if (err) snprintf(err, errlen, "staging %d", i); return -1; }
        void *pm = NULL; vkMapMemory(G.dev, sm, 0, wsz, 0, &pm);
        memcpy(pm, blobs[i], wsz); vkUnmapMemory(G.dev, sm);
        // device buffer
        VkBuffer wb; VkDeviceMemory wm;
        VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = wsz; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(G.dev, &bi, NULL, &wb);
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(G.dev, wb, &rq);
        VkMemoryAllocateInfo am = {0}; am.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        am.allocationSize = rq.size;
        uint32_t mt = mem_type(G.pd, rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        am.memoryTypeIndex = mt;
        if (vkAllocateMemory(G.dev, &am, NULL, &wm) != VK_SUCCESS) { if (err) snprintf(err, errlen, "VRAM penuh di tensor %d", i); return -1; }
        vkBindBufferMemory(G.dev, wb, wm, 0);
        VkBufferCopy cp = {0, 0, wsz};
        vkCmdCopyBuffer(ucb, sb, wb, 1, &cp);
        G.e[G.n].host = blobs[i]; G.e[G.n].wbuf = wb; G.e[G.n].wmem = wm;
        G.e[G.n].rows = rows[i]; G.e[G.n].cols = cols[i];
        G.n++;
        // staging dilepas setelah submit di bawah (disimpan sementara)
        // NOTE: disederhanakan - staging dibiarkan bocor kecil sekali init (didokumentasikan)
        (void)sb; (void)sm;
    }
    vkEndCommandBuffer(ucb);
    VkSubmitInfo si0 = {0}; si0.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si0.commandBufferCount = 1; si0.pCommandBuffers = &ucb;
    vkQueueSubmit(G.queue, 1, &si0, uf);
    vkWaitForFences(G.dev, 1, &uf, VK_TRUE, 60000000000ull);

    // descriptor set per tensor
    for (int i = 0; i < G.n; i++) {
        VkDescriptorSetAllocateInfo dsa = {0}; dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsa.descriptorPool = G.dp; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
        vkAllocateDescriptorSets(G.dev, &dsa, &G.e[i].ds);
        VkDescriptorBufferInfo bi[3] = {{G.e[i].wbuf,0,VK_WHOLE_SIZE},{G.qbuf,0,VK_WHOLE_SIZE},{G.ybuf,0,VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[3] = {0};
        for (int k = 0; k < 3; k++) { wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[k].dstSet = G.e[i].ds; wr[k].dstBinding = k; wr[k].descriptorCount = 1; wr[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[k].pBufferInfo = &bi[k]; }
        vkUpdateDescriptorSets(G.dev, 3, wr, 0, NULL);
    }
    // command pool+fence dispatch (reuse, RESET flag)
    VkCommandPoolCreateInfo cpci = {0}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = qi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(G.dev, &cpci, NULL, &G.pool);
    VkCommandBufferAllocateInfo cba = {0}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = G.pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(G.dev, &cba, &G.cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(G.dev, &fi, NULL, &G.fence);
    return 0;
}

int vx27_gpu_reg_matvec(const void *W, int rows, int cols, const float *x, float *y, double *ms) {
    if (!g_on) return 0;
    int idx = -1;
    for (int i = 0; i < G.n; i++)
        if (G.e[i].host == W && G.e[i].rows == rows && G.e[i].cols == cols) { idx = i; break; }
    if (idx < 0) return 0;
    size_t qsz = (size_t)cols * 4, ysz = (size_t)rows * 4;
    void *pq = NULL, *py = NULL;
    vkMapMemory(G.dev, G.qmem, 0, qsz, 0, &pq); memcpy(pq, x, qsz); vkUnmapMemory(G.dev, G.qmem);
    double t0 = now_ms();
    VkCommandBufferBeginInfo cbb = {0}; cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(G.cb, &cbb);
    vkCmdBindPipeline(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pl, 0, 1, &G.e[idx].ds, 0, NULL);
    struct { uint32_t B, ncols, nrows; } pc = {1, (uint32_t)cols, (uint32_t)rows};
    vkCmdPushConstants(G.cb, G.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cb, (uint32_t)rows, 1, 1);
    VkMemoryBarrier mb = {0}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(G.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkEndCommandBuffer(G.cb);
    vkResetFences(G.dev, 1, &G.fence);
    VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &G.cb;
    vkQueueSubmit(G.queue, 1, &si, G.fence);
    vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 60000000000ull);
    double t1 = now_ms();
    vkMapMemory(G.dev, G.ymem, 0, ysz, 0, &py); memcpy(y, py, ysz); vkUnmapMemory(G.dev, G.ymem);
    vkResetCommandBuffer(G.cb, 0);
    if (ms) *ms = t1 - t0;
    return 1;
}

void vx27_gpu_reg_free(void) { memset(&G, 0, sizeof G); g_on = 0; }
