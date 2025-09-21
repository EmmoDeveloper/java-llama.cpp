# System Hardware Information
Generated on vr 19 sep 2025 15:12:40 CEST

## CPU Information
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 9 5900HX with Radeon Graphics
CPU family:                              25
Model:                                   80
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               1
Stepping:                                0
Frequency boost:                         enabled
CPU(s) scaling MHz:                      73%
CPU max MHz:                             4683,0000
CPU min MHz:                             400,0000
BogoMIPS:                                6587,83
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm debug_swap
Virtualization:                          AMD-V
L1d cache:                               256 KiB (8 instances)
L1i cache:                               256 KiB (8 instances)
L2 cache:                                4 MiB (8 instances)
L3 cache:                                16 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-15
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Mitigation; Safe RET
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsx async abort:           Not affected

## Memory Information
               total        used        free      shared  buff/cache   available
Mem:            31Gi       5,9Gi        17Gi        86Mi       8,5Gi        25Gi
Swap:           48Gi          0B        48Gi

### Memory Details
MemTotal:       32711216 kB
MemFree:        18220712 kB
MemAvailable:   26582548 kB
Buffers:          346748 kB
Cached:          8260200 kB
SwapCached:            0 kB
Active:         10548988 kB
Inactive:        2537520 kB
Active(anon):    4567680 kB
Inactive(anon):        0 kB

## GPU Information
Fri Sep 19 15:13:38 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   54C    P8             15W /  115W |     675MiB /  16384MiB |     14%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2163      G   /usr/lib/xorg/Xorg                      331MiB |
|    0   N/A  N/A            2762      G   /usr/bin/kwalletd6                        3MiB |
|    0   N/A  N/A            3036      G   /usr/bin/ksmserver                        3MiB |
|    0   N/A  N/A            3038      G   /usr/bin/kded6                            3MiB |
|    0   N/A  N/A            3039      G   /usr/bin/kwin_x11                        12MiB |
|    0   N/A  N/A            3063      G   /usr/bin/plasmashell                     54MiB |
|    0   N/A  N/A            3123      G   /usr/bin/kaccess                          3MiB |
|    0   N/A  N/A            3124      G   ...it-kde-authentication-agent-1          3MiB |
|    0   N/A  N/A            3325      G   /usr/bin/kdeconnectd                      3MiB |
|    0   N/A  N/A            3361      G   ...-gnu/libexec/DiscoverNotifier          3MiB |
|    0   N/A  N/A            3420      G   ...ibexec/xdg-desktop-portal-kde          3MiB |
|    0   N/A  N/A            4233      G   ...led --variations-seed-version         33MiB |
|    0   N/A  N/A            5071      G   /usr/bin/konsole                          3MiB |
+-----------------------------------------------------------------------------------------+

## Disk Information
Filesystem                  Size  Used Avail Use% Mounted on
tmpfs                       3,2G  2,4M  3,2G   1% /run
/dev/mapper/vgkubuntu-root  915G  833G   37G  96% /
tmpfs                        16G   31M   16G   1% /dev/shm
efivarfs                    148K  110K   34K  77% /sys/firmware/efi/efivars
tmpfs                       5,0M   12K  5,0M   1% /run/lock
tmpfs                       1,0M     0  1,0M   0% /run/credentials/systemd-journald.service
tmpfs                       1,0M     0  1,0M   0% /run/credentials/systemd-resolved.service
/dev/nvme1n1p1              256M   36M  221M  15% /boot/efi
tmpfs                        16G  9,3M   16G   1% /tmp
tmpfs                       3,2G  5,8M  3,2G   1% /run/user/1000

## System Summary
- CPU: AMD Ryzen 9 5900HX (8 cores, 16 threads)
- RAM: 32GB (25GB available)
- GPU: NVIDIA GeForce RTX 3080 Laptop (16GB VRAM, 675MB used)
- Disk: 915GB total, 37GB free
