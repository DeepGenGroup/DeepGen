import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import re

from plot_utils import *
import numpy as np
import matplotlib.patches as mpatches


devices = ['A100', 'H100']
sys_names = ['flashattn', 'tensorrt', 'our']

def extract(log_dir):
  # name: {device}.{model}.{sys}.log
  # time: [model] avg: 1.2 ms
  fn_pattern = r'(?P<device>[^.]+)\.(?P<model>[^.]+)\.(?P<sys>[^.]+)\.log'
  time_pattern = r'\[.*?\] avg (?P<avg_time>[\d.]+) ms'

  data = {
    "attn": {},
    "gemma2": {},
  }
  for exp in data:
    data[exp ] = {sys: {device: -1 for device in devices} for sys in sys_names}
  
  for root, _, files in os.walk(log_dir):
    for file in files:
      match = re.match(fn_pattern, file)
      assert match
      device = match.group('device').upper()
      model = match.group('model')
      sys = match.group('sys')

      fp = os.path.join(root, file)
      with open(fp, 'r') as f:
        content = f.read()
        avg_match = re.search(time_pattern, content)
        assert avg_match, f"{fp} does not have avg time"
        avg_time = float(avg_match.group('avg_time'))
        data[model][sys][device] = avg_time
  
  attn_df = pd.DataFrame(data['attn'])
  gemma2_df = pd.DataFrame(data['gemma2'])
  return [attn_df, gemma2_df]



def plot(dataset, devices, sys_names, figure_name, add_legend=False):
  # 两图合并，参数有所修改
  figsize = {
      "figure.figsize": (6, 1),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size': 10,
      'legend.fontsize': 10,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, (ax1, ax2) = plt.subplots(1,2)

  # 和utils.py中的COLOR_DEF相同，共7种颜色
  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  # 设置bar width
  width = 0.15
  width_gap = 0.05
  
  # 绘制第一个图
  ylim1 = 2 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  r1 = np.arange(len(devices))
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[0].loc[devices][sys_name]
    bars = ax1.bar(r1, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[devices[i]] > ylim1: 
        # 截断，文字补充
        ax1.text(bar.get_x() + bar.get_width() / 2, ylim1 * 0.95, f'{data_ref[devices[i]]:.2f}', ha='center', va='top', rotation=90)
    r1 = [x + width + width_gap for x in r1]
  # 设置x轴刻度，标签，标题
  ax1.set_xticks([r + ((len(sys_names)-1)* (width + width_gap)) / 2 for r in range(len(devices))], devices)
  max_height = max([max(dataset[0].loc[devices][sys_name]) for sys_name in sys_names])
  ax1.set_ylim(0, ylim1)
  ax1.set_ylabel('Execution Time (ms)', fontsize=10)
  ax1.set_title(f"{MODEL_NAME['attn']}",loc='center')

  # 绘制第二个图
  ylim2 = 2 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  r2 = np.arange(len(devices))
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[1].loc[devices][sys_name]
    bars = ax2.bar(r2, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[devices[i]] > ylim2: 
        # 截断，文字补充
        ax2.text(bar.get_x() + bar.get_width() / 2, ylim2 * 0.95, f'{data_ref[devices[i]]:.2f}', ha='center', va='top', rotation=90)
    r2 = [x + width + width_gap for x in r2]
  ax2.set_xticks([r + ((len(sys_names)-1)* (width + width_gap)) / 2 for r in range(len(devices))], devices)
  max_height = max([max(dataset[1].loc[devices][sys_name]) for sys_name in sys_names])
  ax2.set_ylim(0, ylim2)
  # ax2.set_ylabel('Execution Time (ms)', fontsize=10)
  ax2.set_title(f"{MODEL_NAME['gemma2']}", loc='center')
  
  if add_legend:
    legend = fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.45))

  plt.subplots_adjust(wspace=0.2)
  fig.savefig(f"{figure_name}.pdf", bbox_inches='tight')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot figure 14')
  parser.add_argument('--log_dir', type=str, default='./backup_logs/', help='The directory of log files')
  args = parser.parse_args()
  print(f"{args=}")

  data = extract(args.log_dir)
  plot(data, devices, sys_names, figure_name='fig14', add_legend=True)