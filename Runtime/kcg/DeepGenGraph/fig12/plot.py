import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import re

from plot_utils import *
import numpy as np
import matplotlib.patches as mpatches

model_names = ['h2o', 'roco', 'keyformer', 'snapkv', 'corm', 'attn', 'gemma2']
sys_names = ['torch', 'dynamo', 'tensorrt', 'tvm', 'korch', 'einnet', 'our']

def extract(log_dir):
  # name: {kernel,e2e}.{device}.{model}.{sys}.log
  # time: [model] avg: 1.2 ms
  fn_pattern = r'(?P<exp>[^.]+)\.(?P<device>[^.]+)\.(?P<model>[^.]+)\.(?P<sys>[^.]+)\.log'
  time_pattern = r'\[.*?\] avg (?P<avg_time>[\d.]+) ms'

  data = {
    "kernel": {"a100": {}, "h100": {}},
    "e2e": {"a100": {}, "h100": {}},
  }
  for exp in data:
    for device in data[exp]:
      data[exp][device] = {sys: {model: -1 for model in model_names} for sys in sys_names}

  for root, _, files in os.walk(log_dir):
    for file in files:
      match = re.match(fn_pattern, file)
      assert match
      exp = match.group('exp')
      device = match.group('device').lower()
      model = match.group('model')
      if model == 'kf':
        model = 'keyformer'
      sys = match.group('sys')

      fp = os.path.join(root, file)
      with open(fp, 'r') as f:
        content = f.read()
        avg_match = re.search(time_pattern, content)
        if avg_match:
          avg_time = float(avg_match.group('avg_time'))
          data[exp][device][sys][model] = avg_time
        else:
          if not (sys == 'tvm' and model == 'corm'):
            print(f"Warning: {fp} does not have avg time")
  
  e2e_a100_df = pd.DataFrame(data['e2e']['a100'])
  e2e_h100_df = pd.DataFrame(data['e2e']['h100'])
  kernel_a100_df = pd.DataFrame(data['kernel']['a100'])
  kernel_h100_df = pd.DataFrame(data['kernel']['h100'])

  return [e2e_a100_df, e2e_h100_df], [kernel_a100_df, kernel_h100_df]

def plot(data, devices, model_names, sys_names, figure_name, is_e2e):
  # 两图合并，参数有所修改
  figsize = {
      "figure.figsize": (12, 2),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size':8,
      'legend.fontsize': 10,
      'xtick.labelsize': 10,
      'ytick.labelsize': 9 if is_e2e else 10,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, axs = plt.subplots(2, len(model_names))

  # 和utils.py中的COLOR_DEF相同，共7种颜色
  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  # 用ABCDEF替代7个sys_name
  abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  abc = abc[:len(sys_names)]
  bar_width = 0.8
  bar_gap = 0.2
  ylim = 1000 if is_e2e else 30

  for row_id, (ax_row, dataset) in enumerate(zip(axs, data)):
    for col_id, (ax, model_name) in enumerate(zip(ax_row, model_names)):
      perf_ref = dataset.loc[model_name][sys_names]

      perf = perf_ref.clip(lower=0)
      # 绝对时间
      baseline = perf.loc[sys_names[-1]]
      norm_perf = perf 

      x_pos = np.arange(len(sys_names))
      bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

      for i, bar in enumerate(bars):
        if perf_ref.loc[sys_names[i]] == 0:
          # OOM
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'OOM', ha='center', va='bottom', rotation=90)
        elif perf_ref.loc[sys_names[i]] == -1:
          # 不支持
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'NS', ha='center', va='bottom', rotation=90)
        elif perf_ref.loc[sys_names[i]] == -2:
          # 超时
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'TLE', ha='center', va='bottom', rotation=90)
        elif norm_perf.loc[sys_names[i]] > ylim: 
          # 截断，文字补充
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.95, f'{norm_perf.loc[sys_names[i]]:.1f}', ha='center', va='top', rotation=90)
      
      min_perf = float('inf')
      for i in perf_ref[:-1]:
        if i > 0:
          min_perf = min(min_perf, i)
      # speedup
      ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, bars[-1].get_height(), f'{min_perf / baseline:.1f}\u00D7', fontweight='bold', ha='center', va='bottom', fontsize=7)

      ax.set_xticks(range(len(abc)), abc)
      # 子图标注model_name
      if row_id == 0:
        ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

      if col_id == 0:
        ax.set_ylabel(devices[row_id], fontsize=10)
        ax.yaxis.set_label_coords(-0.5, 0.5)

      max_height = np.nanmax(norm_perf)
      ax.set_ylim(0, ylim)
      ax.set_yticks(range(0, ylim + 1, 250 if is_e2e else 10))

  # 添加legend    
  legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + SYS_NAME[sys_names[i]]) for i in range(len(sys_names))]
  fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
  fig.text(0.085 if is_e2e else 0.09, 0.5, 'Execution Time (ms)', va='center', rotation='vertical', fontsize=10)
  plt.subplots_adjust(hspace=0.5,wspace=0.4 if is_e2e else 0.3)
  fig.savefig(f"fig12_e2e.pdf" if is_e2e else f"fig12_kernel.pdf", bbox_inches='tight')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot figure 12')
  parser.add_argument('--log_dir', type=str, default='./backup_logs/', help='The directory of log files')
  args = parser.parse_args()
  print(f"{args=}")

  e2e_data, kernel_data = extract(args.log_dir) 
  plot(e2e_data, ['A100', 'H100'], model_names, sys_names, figure_name='e2e', is_e2e=True)
  plot(kernel_data, ['A100', 'H100'], model_names, sys_names, figure_name='kernel', is_e2e=False)
