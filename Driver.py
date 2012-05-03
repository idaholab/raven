import os
import linecache
#paths
code_path       = '/Users/crisr/projects/trunk/raven/'   #path where to find the executable
Raven_out_pat   = '/Users/crisr/projects/trunk/raven/'
Final_out_pat   = '/Users/crisr/projects/trunk/raven/'
Raven_input     = '/Users/crisr/projects/trunk/raven/tests/TypPWR_Mult_CoreChannels_test.i'
Raven_option    = '-s'
executable_name = 'RAVEN-opt'
output_root     = 'out_'

n_sampling      = 10
n_PostProcess   = 4
spacing         = 20
spacing_first   = 17
#run RAVEN
for i in range(0,n_sampling):
    string = code_path + executable_name + ' '+ Raven_option + ' ' + Raven_input + '>'+ Raven_out_pat + output_root +str(i)
    os.system(string)


f_PP_out    = open(Final_out_pat+'PP_out','w')
f_steps_out = open(Final_out_pat+'Time_step_out','w')
#f_steps_out.write('sample id         number of time step\n')


sought_string = 'Postprocessor Values:'

for i in range(0,n_sampling):                                            #loop over files
    last_PP_print = 0
    with open(Raven_out_pat + output_root +str(i),'r') as f:
#        string_list = f.readlines()
        n_time_step = 0
        n_line = 0
        for line in f:
            n_line = n_line + 1
            if line.find(sought_string)!=-1:
                n_time_step = n_time_step + 1
                last_PP_print = n_line
    f_steps_out.write('  '+str(i)+'                  '+str(n_time_step)+'\n')
    for ii in range (4,n_time_step+4):
        recorder_string = linecache.getline(code_path+'out_'+str(i), last_PP_print+ii)
        f_PP_out.write(recorder_string[1:spacing_first])
        for iii in range (0,n_PostProcess):
            f_PP_out.write(recorder_string[spacing_first+1+iii*spacing:spacing_first+spacing+iii*spacing])
        f_PP_out.write('\n')

f_steps_out.close()
f_PP_out.close()


quit ()