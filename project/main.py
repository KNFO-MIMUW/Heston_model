import heston_functions as hf
from uploading_data import theta_0, data
from option_class import option

theta = hf.calibration_heston(theta_0)
option_1 = option(theta, data)
print(option_1.eur_option())
print(option_1.asian_option())