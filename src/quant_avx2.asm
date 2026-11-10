; Q4_K x Q8_K, eight tokens per weight row.
; Win64 ABI: rcx = block_q4_k*, rdx = block_q8_k*, r8 = stride (blocks),
;            r9 = nblocks, [rsp+40] = float out[8].
;
; Keeps all eight integer accumulators in ymm registers. Each q8 pair is a
; memory operand of vpmaddubsw, so the hot loop has no explicit loads and no
; accumulator spills.

default rel

global dot_q4_k_q8_k_x8_asm
global dot_q6_k_q8_k_x8_asm

section .rodata align=32
m4b: db 0x0F

section .text

%define PTR7   0
%define OUTP   8
%define SC16   16
%define MN8    48
%define MVEC   56
%define D16F   88
%define M16F   92
%define TOT    96
%define FRAME  288
%define XSAVE  128

%define Q8SZ   292
%define Q4SZ   144

; --- scale decode ---------------------------------------------------------
%macro DEC_LOW 1
    movzx r9d, byte [rbx + 4 + %1]
    and   r9d, 63
    mov   [rsp + SC16 + 2*%1], r9w
    movzx r9d, byte [rbx + 8 + %1]
    and   r9d, 63
    mov   [rsp + MN8 + %1], r9b
%endmacro

%macro DEC_HIGH 1                    ; j = 4..7
    movzx r9d,  byte [rbx + 4 + %1 + 4]
    mov   r11d, r9d
    and   r9d, 15
    movzx r10d, byte [rbx + 4 + (%1 - 4)]
    shr   r10d, 6
    shl   r10d, 4
    or    r9d, r10d
    mov   [rsp + SC16 + 2*%1], r9w
    shr   r11d, 4
    movzx r10d, byte [rbx + 4 + %1]
    shr   r10d, 6
    shl   r10d, 4
    or    r11d, r10d
    mov   [rsp + MN8 + %1], r11b
%endmacro

; --- inner dot ------------------------------------------------------------
%macro Q4_TOK 3                      ; j, q8 pointer, acc
    vpmaddubsw ymm12, ymm9, [%2 + 4 + 64*%1]
    vpmaddubsw ymm13, ymm8, [%2 + 4 + 64*%1 + 32]
    vpmaddwd   ymm12, ymm12, ymm10
    vpmaddwd   ymm13, ymm13, ymm11
    vpaddd     ymm12, ymm12, ymm13
    vpaddd     %3, %3, ymm12
%endmacro

%macro INNER_J 1
    vmovdqu       ymm8, [rbx + 16 + 32*%1]
    vpand         ymm9, ymm8, ymm14
    vpsrlw        ymm8, ymm8, 4
    vpand         ymm8, ymm8, ymm14
    vpbroadcastw  ymm10, word [rsp + SC16 + 4*%1]
    vpbroadcastw  ymm11, word [rsp + SC16 + 4*%1 + 2]
    Q4_TOK %1, rax, ymm0
    Q4_TOK %1, rcx, ymm1
    Q4_TOK %1, rdx, ymm2
    Q4_TOK %1, rsi, ymm3
    Q4_TOK %1, rdi, ymm4
    Q4_TOK %1, rbp, ymm5
    Q4_TOK %1, r8,  ymm6
    Q4_TOK %1, r9,  ymm7
%endmacro

; --- per-block final ------------------------------------------------------
%macro FINAL_T 4                     ; q8 ptr, ymm acc, xmm acc, index
    vextracti128 xmm8, %2, 1
    vpaddd       xmm8, %3, xmm8
    vphaddd      xmm8, xmm8, xmm8
    vphaddd      xmm8, xmm8, xmm8
    vmovd        r10d, xmm8
    vpmaddwd     ymm8, ymm14, [%1 + 260]
    vextracti128 xmm9, ymm8, 1
    vpaddd       xmm8, xmm8, xmm9
    vphaddd      xmm8, xmm8, xmm8
    vphaddd      xmm8, xmm8, xmm8
    vmovd        r11d, xmm8
    vcvtsi2ss    xmm10, xmm10, r10d
    vcvtsi2ss    xmm11, xmm11, r11d
    vmulss       xmm10, xmm10, dword [rsp + D16F]
    vmulss       xmm11, xmm11, dword [rsp + M16F]
    vsubss       xmm10, xmm10, xmm11
    vmulss       xmm10, xmm10, dword [%1]
    vaddss       xmm10, xmm10, dword [rsp + TOT + 4*%4]
    vmovss       [rsp + TOT + 4*%4], xmm10
%endmacro

dot_q4_k_q8_k_x8_asm:
    mov     r10, [rsp + 40]          ; out (volatile until saved)
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, FRAME
    vmovups [rsp + XSAVE + 0],  xmm6
    vmovups [rsp + XSAVE + 16], xmm7
    vmovups [rsp + XSAVE + 32], xmm8
    vmovups [rsp + XSAVE + 48], xmm9
    vmovups [rsp + XSAVE + 64], xmm10
    vmovups [rsp + XSAVE + 80], xmm11
    vmovups [rsp + XSAVE + 96], xmm12
    vmovups [rsp + XSAVE + 112], xmm13
    vmovups [rsp + XSAVE + 128], xmm14
    vmovups [rsp + XSAVE + 144], xmm15

    mov     rbx, rcx                 ; block cursor
    mov     r12, rdx                 ; q8 base
    mov     r13, r8
    imul    r13, Q8SZ                ; stride bytes
    mov     r14, r9                  ; block count
    mov     [rsp + OUTP], r10

    ; per-token q8 pointers
    mov     rax, r12
    lea     rcx, [r12 + r13]
    lea     rdx, [rcx + r13]
    lea     rsi, [rdx + r13]
    lea     rdi, [rsi + r13]
    lea     rbp, [rdi + r13]
    lea     r8,  [rbp + r13]
    lea     r9,  [r8 + r13]
    mov     [rsp + PTR7], r9         ; token 7 pointer for next block

    vpxor   ymm0, ymm0, ymm0
    vmovdqu [rsp + TOT], ymm0
    vpxor   ymm0, ymm0, ymm0

    test    r14, r14
    jz      .done

.block:
    ; d and dmin (fp16 -> fp32)
    vcvtph2ps xmm0, qword [rbx]
    vmovss    [rsp + D16F], xmm0
    vcvtph2ps xmm0, qword [rbx + 2]
    vmovss    [rsp + M16F], xmm0

    DEC_LOW 0
    DEC_LOW 1
    DEC_LOW 2
    DEC_LOW 3
    DEC_HIGH 4
    DEC_HIGH 5
    DEC_HIGH 6
    DEC_HIGH 7

    ; mvec = [mn0,mn0,mn1,mn1,...,mn7,mn7] as int16
    vmovq      xmm0, qword [rsp + MN8]
    vpunpcklbw xmm0, xmm0, xmm0
    vpmovzxbw  ymm14, xmm0
    vmovdqu    [rsp + MVEC], ymm14

    vpbroadcastb ymm14, byte [m4b]
    mov     r9, [rsp + PTR7]

    vpxor   ymm0, ymm0, ymm0
    vpxor   ymm1, ymm1, ymm1
    vpxor   ymm2, ymm2, ymm2
    vpxor   ymm3, ymm3, ymm3
    vpxor   ymm4, ymm4, ymm4
    vpxor   ymm5, ymm5, ymm5
    vpxor   ymm6, ymm6, ymm6
    vpxor   ymm7, ymm7, ymm7

    INNER_J 0
    INNER_J 1
    INNER_J 2
    INNER_J 3

    vmovdqu ymm14, [rsp + MVEC]

    FINAL_T rax, ymm0, xmm0, 0
    FINAL_T rcx, ymm1, xmm1, 1
    FINAL_T rdx, ymm2, xmm2, 2
    FINAL_T rsi, ymm3, xmm3, 3
    FINAL_T rdi, ymm4, xmm4, 4
    FINAL_T rbp, ymm5, xmm5, 5
    FINAL_T r8,  ymm6, xmm6, 6
    FINAL_T r9,  ymm7, xmm7, 7

    add     rax, Q8SZ
    add     rcx, Q8SZ
    add     rdx, Q8SZ
    add     rsi, Q8SZ
    add     rdi, Q8SZ
    add     rbp, Q8SZ
    add     r8,  Q8SZ
    add     r9,  Q8SZ
    mov     [rsp + PTR7], r9

    add     rbx, Q4SZ
    dec     r14
    jnz     .block

.done:
    vmovdqu ymm0, [rsp + TOT]
    mov     r15, [rsp + OUTP]
    vmovups [r15], ymm0
    vzeroupper
    vmovups xmm6,  [rsp + XSAVE + 0]
    vmovups xmm7,  [rsp + XSAVE + 16]
    vmovups xmm8,  [rsp + XSAVE + 32]
    vmovups xmm9,  [rsp + XSAVE + 48]
    vmovups xmm10, [rsp + XSAVE + 64]
    vmovups xmm11, [rsp + XSAVE + 80]
    vmovups xmm12, [rsp + XSAVE + 96]
    vmovups xmm13, [rsp + XSAVE + 112]
    vmovups xmm14, [rsp + XSAVE + 128]
    vmovups xmm15, [rsp + XSAVE + 144]
    add     rsp, FRAME
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret

; ---------------------------------------------------------------------------
; Q6_K x Q8_K, eight tokens per weight row.
; Win64 ABI: rcx = block_q6_k*, rdx = block_q8_k*, r8 = stride (blocks),
;            r9 = nblocks, [rsp+40] = float out[8].
; ---------------------------------------------------------------------------

%define Q4S     0
%define SCS     128
%define Q8SUB   256
%define DF      512
%define ACC     544
%define OUTP    800
%define DBASE   808
%define XSAVE6  816
%define FRAME6  976

%define Q6SZ    210

%macro Q6_TOK 3                      ; q8 ptr, sumi, qs offset
    vmovdqu      ymm8, [rsp + Q4S + 0]
    vpmaddubsw   ymm8, ymm8, [%1 + 4 + %3 + 0]
    vpmaddwd     ymm8, ymm8, [rsp + SCS + 0]
    vpaddd       %2, %2, ymm8
    vmovdqu      ymm8, [rsp + Q4S + 32]
    vpmaddubsw   ymm8, ymm8, [%1 + 4 + %3 + 32]
    vpmaddwd     ymm8, ymm8, [rsp + SCS + 32]
    vpaddd       %2, %2, ymm8
    vmovdqu      ymm8, [rsp + Q4S + 64]
    vpmaddubsw   ymm8, ymm8, [%1 + 4 + %3 + 64]
    vpmaddwd     ymm8, ymm8, [rsp + SCS + 64]
    vpaddd       %2, %2, ymm8
    vmovdqu      ymm8, [rsp + Q4S + 96]
    vpmaddubsw   ymm8, ymm8, [%1 + 4 + %3 + 96]
    vpmaddwd     ymm8, ymm8, [rsp + SCS + 96]
    vpaddd       %2, %2, ymm8
%endmacro

%macro Q6_SETUP_D 2                  ; token ptr, index
    vmovss       xmm0, dword [%1]
    vmulss       xmm0, xmm0, dword [rsp + DBASE]
    vmovss       [rsp + DF + 4*%2], xmm0
%endmacro

%macro Q6_SETUP_SUB 2                ; token ptr, index
    vmovdqu      ymm0, [%1 + 260]
    vpmaddwd     ymm0, ymm0, ymm15
    vpslld       ymm0, ymm0, 5
    vmovdqu      [rsp + Q8SUB + 32*%2], ymm0
%endmacro

%macro Q6_ACCBLK 2                   ; index, sumi
    vmovdqu      ymm8, [rsp + Q8SUB + 32*%1]
    vpsubd       ymm8, %2, ymm8
    vcvtdq2ps    ymm8, ymm8
    vbroadcastss ymm9, dword [rsp + DF + 4*%1]
    vfmadd213ps  ymm8, ymm9, [rsp + ACC + 32*%1]
    vmovdqu      [rsp + ACC + 32*%1], ymm8
%endmacro

%macro Q6_HSUM 1
    vmovdqu      ymm0, [rsp + ACC + 32*%1]
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vhaddps      xmm0, xmm0, xmm0
    vhaddps      xmm0, xmm0, xmm0
    vmovss       [r10 + 4*%1], xmm0
%endmacro

dot_q6_k_q8_k_x8_asm:
    mov     r10, [rsp + 40]          ; out
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, FRAME6
    vmovups [rsp + XSAVE6 + 0],  xmm6
    vmovups [rsp + XSAVE6 + 16], xmm7
    vmovups [rsp + XSAVE6 + 32], xmm8
    vmovups [rsp + XSAVE6 + 48], xmm9
    vmovups [rsp + XSAVE6 + 64], xmm10
    vmovups [rsp + XSAVE6 + 80], xmm11
    vmovups [rsp + XSAVE6 + 96], xmm12
    vmovups [rsp + XSAVE6 + 112], xmm13
    vmovups [rsp + XSAVE6 + 128], xmm14
    vmovups [rsp + XSAVE6 + 144], xmm15
    vpxor   ymm0, ymm0, ymm0
    vmovdqu [rsp + ACC + 0], ymm0
    vmovdqu [rsp + ACC + 32], ymm0
    vmovdqu [rsp + ACC + 64], ymm0
    vmovdqu [rsp + ACC + 96], ymm0
    vmovdqu [rsp + ACC + 128], ymm0
    vmovdqu [rsp + ACC + 160], ymm0
    vmovdqu [rsp + ACC + 192], ymm0
    vmovdqu [rsp + ACC + 224], ymm0

    mov     rbx, rcx
    mov     r12, rdx
    mov     r13, r8
    imul    r13, Q8SZ
    mov     r14, r9
    mov     [rsp + OUTP], r10

    mov     rax, r12
    lea     rcx, [r12 + r13]
    lea     rdx, [rcx + r13]
    lea     rsi, [rdx + r13]
    lea     rdi, [rsi + r13]
    lea     rbp, [rdi + r13]
    lea     r8,  [rbp + r13]
    lea     r9,  [r8 + r13]

    test    r14, r14
    jz      .done

.block:
    movzx   r10d, word [rbx + 208]
    vmovd   xmm0, r10d
    vcvtph2ps xmm0, xmm0
    vmovss  [rsp + DBASE], xmm0

    vmovdqu xmm14, [rbx + 192]       ; scales[16] as int8
    vpmovsxbw ymm15, xmm14           ; scales as int16

    Q6_SETUP_D rax, 0
    Q6_SETUP_D rcx, 1
    Q6_SETUP_D rdx, 2
    Q6_SETUP_D rsi, 3
    Q6_SETUP_D rdi, 4
    Q6_SETUP_D rbp, 5
    Q6_SETUP_D r8,  6
    Q6_SETUP_D r9,  7
    Q6_SETUP_SUB rax, 0
    Q6_SETUP_SUB rcx, 1
    Q6_SETUP_SUB rdx, 2
    Q6_SETUP_SUB rsi, 3
    Q6_SETUP_SUB rdi, 4
    Q6_SETUP_SUB rbp, 5
    Q6_SETUP_SUB r8,  6
    Q6_SETUP_SUB r9,  7

    vpxor   ymm0, ymm0, ymm0
    vpxor   ymm1, ymm1, ymm1
    vpxor   ymm2, ymm2, ymm2
    vpxor   ymm3, ymm3, ymm3
    vpxor   ymm4, ymm4, ymm4
    vpxor   ymm5, ymm5, ymm5
    vpxor   ymm6, ymm6, ymm6
    vpxor   ymm7, ymm7, ymm7

%assign HALF 0
%rep 2
    ; q4 fragments
    vmovdqu ymm8, [rbx + 64*HALF]
    vmovdqu ymm9, [rbx + 64*HALF + 32]
    vmovdqu ymm10, [rbx + 128 + 32*HALF]
    vpand   ymm11, ymm10, [rel q6_m3]
    vpsllw  ymm11, ymm11, 4
    vpand   ymm12, ymm8, [rel q6_m15]
    vpor    ymm12, ymm12, ymm11
    vmovdqu [rsp + Q4S + 0], ymm12
    vpand   ymm11, ymm10, [rel q6_m12]
    vpsllw  ymm11, ymm11, 2
    vpand   ymm13, ymm9, [rel q6_m15]
    vpor    ymm13, ymm13, ymm11
    vmovdqu [rsp + Q4S + 32], ymm13
    vpand   ymm11, ymm10, [rel q6_m48]
    vpsrlw  ymm12, ymm8, 4
    vpand   ymm12, ymm12, [rel q6_m15]
    vpor    ymm12, ymm12, ymm11
    vmovdqu [rsp + Q4S + 64], ymm12
    vpand   ymm11, ymm10, [rel q6_m192]
    vpsrlw  ymm11, ymm11, 2
    vpsrlw  ymm12, ymm9, 4
    vpand   ymm12, ymm12, [rel q6_m15]
    vpor    ymm12, ymm12, ymm11
    vmovdqu [rsp + Q4S + 96], ymm12

    ; scales for this half: 2*(is+k) repeated per 16-bit lane pair
    vpshufb xmm8, xmm14, [rel q6_shuf + 16*(4*HALF + 0)]
    vpmovsxbw ymm8, xmm8
    vmovdqu [rsp + SCS + 0], ymm8
    vpshufb xmm8, xmm14, [rel q6_shuf + 16*(4*HALF + 1)]
    vpmovsxbw ymm8, xmm8
    vmovdqu [rsp + SCS + 32], ymm8
    vpshufb xmm8, xmm14, [rel q6_shuf + 16*(4*HALF + 2)]
    vpmovsxbw ymm8, xmm8
    vmovdqu [rsp + SCS + 64], ymm8
    vpshufb xmm8, xmm14, [rel q6_shuf + 16*(4*HALF + 3)]
    vpmovsxbw ymm8, xmm8
    vmovdqu [rsp + SCS + 96], ymm8

    Q6_TOK rax, ymm0, 128*HALF
    Q6_TOK rcx, ymm1, 128*HALF
    Q6_TOK rdx, ymm2, 128*HALF
    Q6_TOK rsi, ymm3, 128*HALF
    Q6_TOK rdi, ymm4, 128*HALF
    Q6_TOK rbp, ymm5, 128*HALF
    Q6_TOK r8,  ymm6, 128*HALF
    Q6_TOK r9,  ymm7, 128*HALF
%assign HALF HALF+1
%endrep

    Q6_ACCBLK 0, ymm0
    Q6_ACCBLK 1, ymm1
    Q6_ACCBLK 2, ymm2
    Q6_ACCBLK 3, ymm3
    Q6_ACCBLK 4, ymm4
    Q6_ACCBLK 5, ymm5
    Q6_ACCBLK 6, ymm6
    Q6_ACCBLK 7, ymm7

    add     rax, Q8SZ
    add     rcx, Q8SZ
    add     rdx, Q8SZ
    add     rsi, Q8SZ
    add     rdi, Q8SZ
    add     rbp, Q8SZ
    add     r8,  Q8SZ
    add     r9,  Q8SZ

    add     rbx, Q6SZ
    dec     r14
    jnz     .block

.done:
    mov     r10, [rsp + OUTP]
    Q6_HSUM 0
    Q6_HSUM 1
    Q6_HSUM 2
    Q6_HSUM 3
    Q6_HSUM 4
    Q6_HSUM 5
    Q6_HSUM 6
    Q6_HSUM 7
    vzeroupper
    vmovups xmm6,  [rsp + XSAVE6 + 0]
    vmovups xmm7,  [rsp + XSAVE6 + 16]
    vmovups xmm8,  [rsp + XSAVE6 + 32]
    vmovups xmm9,  [rsp + XSAVE6 + 48]
    vmovups xmm10, [rsp + XSAVE6 + 64]
    vmovups xmm11, [rsp + XSAVE6 + 80]
    vmovups xmm12, [rsp + XSAVE6 + 96]
    vmovups xmm13, [rsp + XSAVE6 + 112]
    vmovups xmm14, [rsp + XSAVE6 + 128]
    vmovups xmm15, [rsp + XSAVE6 + 144]
    add     rsp, FRAME6
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret

section .rodata align=32
q6_m3:   times 32 db 3
q6_m12:  times 32 db 12
q6_m15:  times 32 db 15
q6_m48:  times 32 db 48
q6_m192: times 32 db 0xC0
align 16
q6_shuf:
    db 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
    db 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    db 4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5
    db 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
    db 8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9
    db 10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11
    db 12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13
    db 14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
