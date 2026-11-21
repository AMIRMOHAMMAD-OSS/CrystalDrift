# Measured capacity and exposure

Each row uses 96 development targets to select the checkpoint and a disjoint 96-target confirmation panel. The identical prior recovered 59 development and 48 confirmation targets at five draws. Native volume requires a match within a factor of 1.1 of the target volume on that same proposal. Training time includes checkpoint writes and validation geometry generation and excludes CPU structure matching.

| Fit | Deployed weights | Extra training weights | Updates | First-pass seconds | Peak CUDA MiB | Selected | Development | Confirmation | Native confirmation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| tiny-angular-laplace-11-published | 6,723 | 0 | 1,652 | 134.0 | 42.7 | half | 63 | 50 | 26 |
| small-angular-laplace-11-published | 218,883 | 0 | 1,652 | 210.2 | 148.7 | full | 62 | 52 | 27 |
| small-angular-laplace-11-local | 179,509 | 0 | 1,652 | 169.9 | 50.9 | half | 63 | 50 | 31 |
| small-angular-laplace-11 | 218,497 | 0 | 1,652 | 120.2 | 50.3 | half | 63 | 50 | 31 |
| medium-angular-laplace-11 | 2,098,297 | 0 | 1,652 | 169.2 | 110.9 | half | 62 | 51 | 33 |
| medium-contacts-laplace-11 | 2,098,297 | 0 | 1,652 | 173.1 | 113.6 | half | 63 | 50 | 33 |
| large-angular-laplace-11 | 15,011,747 | 0 | 1,652 | 572.3 | 405.8 | half | 62 | 50 | 35 |
| small-angular-laplace-11-systemaux | 218,497 | 805 | 1,652 | 169.4 | 50.4 | half | 63 | 50 | 29 |
| reference-contacts-laplace-11 | 135,992,027 | 0 | 13,210 | 4340.7 | 2631.5 | half | 62 | 50 | 13 |

## Comparator-sized profile

The 135,992,027-parameter contacts model completed 64 measured optimizer updates on the local RTX 3060 Laptop GPU. After one warm-up update, the mean was 0.191 s and p95 was 0.277 s. The measured peak was 2631.5 MiB. At microbatch 2 a traversal of 26,419 anchors takes 13,210 updates, compared with 1,652 at microbatch 16. Extrapolating the warm mean gives 0.70 GPU hours before checkpoint writes and selection.

The 15M row extends the fixed-angular, microbatch-16 series. The comparator-sized contact row uses the promoted descriptor and about eight times as many optimizer updates per traversal. The medium contact row supplies its descriptor-matched reference at matched data exposure. The auxiliary row changes supervision at the original small angular capacity and inference interface.

The comparator-sized full fit occupied 1.36 wall hours from its positive capacity gate to its final fit and optimizer receipt, including a checkpoint restart, I/O and validation structure generation.
