import logging
import os, sys, warnings
from dataclasses import dataclass
import datetime



# logs main folder
log_folder_name = 'logs'
os.makedirs(os.path.join(os.getcwd(), log_folder_name), exist_ok =True)

log_main_folder_path = os.getcwd() + '/logs'

# logs sub folder
sub_folder_date = datetime.datetime.now().strftime("%Y_%m_%d")
log_sub_folder_name = os.path.join(log_main_folder_path, sub_folder_date)
os.makedirs(log_sub_folder_name, exist_ok =True)

print(log_sub_folder_name)


# log file path
file_date = str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))+ ".log"
print(file_date)
log_file_path = os.path.join(log_main_folder_path,log_sub_folder_name, file_date)

logging.basicConfig(filename = log_file_path,
                    level = logging.INFO,
                    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")


# # test
# if __name__ == '__main__':
#     try:
#         3/0
#     except Exception as e:
#         logging.info(e