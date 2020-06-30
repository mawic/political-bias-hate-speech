import subprocess

folds_per_run="3"
runs="1"
steps_data_exchange="3"

#left
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_1" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_2" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_3" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_4" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_5" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_6" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_7" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_8" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_9" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/left/LEFT_10" -d "LEFT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> LEFT.log &', shell=True)

#right
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_1" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_2" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_3" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_4" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_5" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_6" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_7" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_8" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_9" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/right/RIGHT_10" -d "RIGHT" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> RIGHT.log &', shell=True)

#neutral
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_1" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_2" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_3" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_4" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_5" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_6" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_7" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_8" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_9" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
subprocess.check_output('python run_partly_data_exchange.py -o "output/classifier/neutral/NEUTRAL_10" -d "NEUTRAL" -f '+folds_per_run+' -r '+runs+' -s '+steps_data_exchange+' 2> NEUTRAL.log &', shell=True)
