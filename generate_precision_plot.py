# -- Generating Supplement Figure S2 -- precision of the evaluating partition function

import matplotlib.pyplot as plt
from precision_estimate import plot_entropy_gap_dependence,plot_error_gap_dependence,plot_error_precision_dependence

plt.figure(figsize=(10, 4))
ax1,ax2,ax3 = plt.subplot(131),plt.subplot(132),plt.subplot(133)

# generate the dependence of entropy as function of the gap
plot_entropy_gap_dependence(ax=ax1,nu=0.,system_size = [16,24,32,40],overwrite = False)

# generate the dependence of entropy ERROR as function of the gap
# a0 is the scaling index (slope of the dotted lines)
a0 = plot_error_gap_dependence(ax=ax2,nu=0.,system_size = [16,24,32,40],overwrite = False)

# generate the enrtopy error as function of digital precision
plot_error_precision_dependence(ax=ax3,nu=0.,system_size = [16,24,32,40],a0=a0,overwrite = False)

plt.tight_layout()
plt.show()
