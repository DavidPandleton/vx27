#include "vx27_gpu_dl.h"
#include <vulkan/vulkan.h>
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
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0;
}

int vx27_gpu_dl_upload(size_t bytes, double *ms, double *bw_gbs, char *err, size_t errlen) {
    (void)err; (void)errlen;
    VkApplicationInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &ai;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return -1;
    uint32_t nd = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &nd, &pd);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; while (qi < nq && !(qf[qi].queueFlags & VK_QUEUE_TRANSFER_BIT)) qi++;
    if (qi == nq) qi = 0;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { free(qf); return -1; }
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);

    VkBuffer sbuf, dbuf; VkDeviceMemory smem, dmem;
    VkBufferCreateInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &bi, NULL, &sbuf);
    VkMemoryRequirements r1; vkGetBufferMemoryRequirements(dev, sbuf, &r1);
    VkMemoryAllocateInfo a1 = {0}; a1.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    a1.allocationSize = r1.size;
    a1.memoryTypeIndex = mem_type(pd, r1.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(dev, &a1, NULL, &smem); vkBindBufferMemory(dev, sbuf, smem, 0);
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkCreateBuffer(dev, &bi, NULL, &dbuf);
    VkMemoryRequirements r2; vkGetBufferMemoryRequirements(dev, dbuf, &r2);
    VkMemoryAllocateInfo a2 = {0}; a2.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    a2.allocationSize = r2.size;
    // prefer DEVICE_LOCAL
    uint32_t dt = 0; VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((r2.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { dt = i; break; }
    a2.memoryTypeIndex = dt;
    vkAllocateMemory(dev, &a2, NULL, &dmem); vkBindBufferMemory(dev, dbuf, dmem, 0);

    void *p = NULL; vkMapMemory(dev, smem, 0, bytes, 0, &p); memset(p, 0xAB, bytes); vkUnmapMemory(dev, smem);

    VkCommandPoolCreateInfo cp = {0}; cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cp.queueFamilyIndex = qi;
    VkCommandPool pool; vkCreateCommandPool(dev, &cp, NULL, &pool);
    VkCommandBufferAllocateInfo ca = {0}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &ca, &cb);
    VkFenceCreateInfo fi = {0}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; vkCreateFence(dev, &fi, NULL, &fence);

    double t0 = now_ms();
    VkCommandBufferBeginInfo bb = {0}; bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(dev == 0 ? 0 : cb, &bb);
    VkBufferCopy cp2 = {0, 0, bytes};
    vkCmdCopyBuffer(cb, sbuf, dbuf, 1, &cp2);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(q, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, 60000000000ull);
    double t1 = now_ms();
    if (ms) *ms = t1 - t0;
    if (bw_gbs) *bw_gbs = (double)bytes / ((t1 - t0) / 1000.0) / 1e9;
    free(qf);
    return 0;
}
