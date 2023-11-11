#!/usr/bin/python3 -u

import os
import sys

source = '0'
exe_name = 'minibenchmark'

def compile_code(code_path, code, programming_model):
    print("compiling %s" % code)
    if programming_model == '2':
        os.system('nvcc %s -O3 -arch=sm_70 -Iinclude -o %s' % (code_path + code, exe_name))
    elif programming_model == '1':
        os.system('g++ %s -O3 -fopenmp -Iinclude -o %s' % (code_path + code, exe_name))
    elif programming_model == '0':
        os.system('g++ %s -O3 -pthread -Iinclude -o %s' % (code_path + code, exe_name))
    else:
        sys.exit('Error: invalid programming model\n0: c++ threads\n1: openmp\n2: cuda')

def run_code(input_path, input_files, code_path, programming_model, verify, thread_count):
    # run through every input
    counter = 0
    for input in input_files:

        # print progress
        counter += 1
        print("%s: %d out of %d inputs" % (input, counter, len(input_files)))

        # run test
        if programming_model == '2':
            if 'sssp' in code_path or 'bfs' in code_path:
                os.system('./%s %s %s %s' % (exe_name, input_path + input, source, verify))
            elif 'pr' in code_path:
                os.system('./%s %s' % (exe_name, input_path + input))
            else:
                os.system('./%s %s %s' % (exe_name, input_path + input, verify))
        elif programming_model == '1':
            os.system('export OMP_NUM_THREADS=%s' % thread_count)
            if 'sssp' in code_path or 'bfs' in code_path:
                os.system('./%s %s %s %s' % (exe_name, input_path + input, source, verify))
            elif 'pr' in code_path:
                os.system('./%s %s' % (exe_name, input_path + input))
            else:
                os.system('./%s %s %s' % (exe_name, input_path + input, verify))        
        elif programming_model == '0':
            if 'sssp' in code_path or 'bfs' in code_path:
                os.system('./%s %s %s %s %s' % (exe_name, input_path + input, source, verify, thread_count))
            elif 'pr' in code_path:
                os.system('./%s %s %s' % (exe_name, input_path + input, thread_count))
            else:
                os.system('./%s %s %s %s' % (exe_name, input_path + input, verify, thread_count))        
        else:
            sys.exit('Error: invalid programming model\n0: c++ threads\n1: openmp\n2: cuda')
        sys.stdout.flush()

    # delete executable
    if os.path.isfile(exe_name):
        os.system('rm %s' % exe_name)
    else:
        sys.exit('Error: compile failed')

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) < 4):
        sys.exit('USAGE: code_directory input_directory programming_model verify(optional) thread_count(optional)\n')
    code_path = args_val[1]
    input_path = args_val[2]
    programming_model = args_val[3]
    verify = 1
    thread_count = 16
    if (len(args_val) > 4):
        verify = args_val[4]
    if (len(args_val) > 5):
        thread_count = args_val[5]

    # list code files
    code_files = [f for f in os.listdir(code_path) if os.path.isfile(os.path.join(code_path, f))]
    num_codes = len(code_files)

    # list input files
    input_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    num_inputs = len(input_files)
    print("num_codes: %d\nnum_inputs: %d\n" % (num_codes, num_inputs))
    print("code_path: %s\ninput_path: %s\n" % (code_path, input_path))

    for code in code_files:
        compile_code(code_path, code, programming_model)
        run_code(input_path, input_files, code_path, programming_model, verify, thread_count)
