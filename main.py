import json
import shield
import os

filename = 'SwitchShieldx'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with open('cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
for c_i in load_file:
   for cfg in load_file[c_i]:
      # cfg['fixed_policy'] = True
      # cfg['fixed_policy_p'] = 0.25
      cfg['video_path'] = 'newvideos/NS/' + cfg['video_path'][10:]
      # if cfg['noshield']:
      shield.main(cfg)

