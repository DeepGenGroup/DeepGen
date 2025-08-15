import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import re

from plot_utils import *
import numpy as np
import matplotlib.patches as mpatches

seqlens = [1024, 2048, 4096, 8192]
sys_names = ['torch', 'dynamo', 'tensorrt', 'our']

def extract(log_dir):
  # name: {device}.{sys}.{seqlen}.log
  fn_pattern = r'(?P<device>[^.]+)\.(?P<sys>[^.]+)\.(?P<seqlen>[^.]+)\.log'
  gflops_pattern = r'(?P<gflops>[\d.]+)\s*gflops/s'
  mib_pattern = r'\nmib=(?P<mib>[\d.]+)'

  data = {
    "gflops": {"h100": {}},
    "mem": {"h100": {}},
  }
  for metric in data:
    for device in data[metric]:
      data[metric][device] = {sys: {seqlen: -1 for seqlen in seqlens} for sys in sys_names}
  
  for root, _, files in os.walk(log_dir):
    for file in files:
      match = re.match(fn_pattern, file)
      assert match
      device = match.group('device').lower()
      sys = match.group('sys')
      seqlen = int(match.group('seqlen'))

      fp = os.path.join(root, file)
      with open(fp, 'r') as f:
        content = f.read()
        gflops_match = re.search(gflops_pattern, content)
        assert gflops_match, f"{fp} does not have gflops"
        gflops = float(gflops_match.group('gflops'))

        mib_match = re.search(mib_pattern, content)
        assert mib_match, f"{fp} does not have mib"
        mib = float(mib_match.group('mib'))
        if sys == 'tensorrt':
          activation_memory = re.search(r'Total Activation Memory:\s*(\d+) bytes', content)
          weights_memory = re.search(r'Total Weights Memory:\s*(\d+) bytes', content)
          assert activation_memory, f"{fp} does not have activation memory"
          assert weights_memory, f"{fp} does not have weights memory"
          mib = (int(activation_memory.group(1)) + int(weights_memory.group(1))) / 1024 / 1024
        
        data['gflops'][device][sys][seqlen] = gflops
        data['mem'][device][sys][seqlen] = mib
  
  tflops_df = pd.DataFrame(data['gflops']['h100']) / 1000
  mem_df = pd.DataFrame(data['mem']['h100'])

  return [tflops_df, mem_df]

def plot(dataset, seqlens, sys_names, figure_name, add_legend=False):
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

  # 设置bar width gap
  width = 0.16
  width_gap = 0.04

  # 绘制第一个图
  ylim1 = 125 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  r1 = np.arange(len(seqlens))
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[0].loc[seqlens][sys_name]
    bars = ax1.bar(r1, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[seqlens[i]] > ylim1: 
        # 截断，文字补充
        ax1.text(bar.get_x() + bar.get_width() / 2, ylim1 * 0.95, f'{data_ref[seqlens[i]]:.2f}', ha='center', va='top', rotation=90)
    r1 = [x + width + width_gap for x in r1]
  # 设置x轴刻度，标签，标题
  ax1.set_xticks([r + ((len(sys_names)-1) * (width + width_gap)) / 2 for r in range(len(seqlens))], seqlens)
  ax1.set_xlabel('Sequence Length', fontsize=10)
  max_height = max([max(dataset[0].loc[seqlens][sys_name]) for sys_name in sys_names])
  ax1.set_ylim(0, ylim1)
  ax1.set_ylabel('TFLOP/s', fontsize=10)
  ax1.set_title("Core Module TFLOP/s",loc='center')
  
  # 绘制第二个图
  r2 = np.arange(len(seqlens))
  ylim2 = 10 ** 5 # 限制y轴范围, 保持表格整齐，可根据数据手动调整
  for i, sys_name in enumerate(sys_names):
    data_ref = dataset[1].loc[seqlens][sys_name]
    bars = ax2.bar(r2, data_ref, hatch=hatch_def[i], color=pair_color_def[i], width=width, label=SYS_NAME[sys_name], edgecolor='k')
    for i, bar in enumerate(bars):
      if data_ref[seqlens[i]] > ylim2: 
        # 截断，文字补充
        ax2.text(bar.get_x() + bar.get_width() / 2, ylim2 * 0.95, f'{data_ref[seqlens[i]]:.2f}', ha='center', va='top', rotation=90)
    r2 = [x + width + width_gap for x in r2]
  ax2.set_xticks([r + ((len(sys_names)-1) * (width + width_gap)) / 2 for r in range(len(seqlens))], seqlens)
  ax2.set_xlabel('Sequence Length', fontsize=10)
  max_height = max([max(dataset[1].loc[seqlens][sys_name]) for sys_name in sys_names])
  ax2.set_yscale('log', base=10)
  ax2.set_ylim(100, ylim2)
  ax2.set_ylabel('Memory Usage (MiB)', fontsize=10)
  ax2.set_title("Core Module Memory Usage", loc='center')

  if add_legend:
    legend = fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.45))

  # 调整子图之间的间距
  plt.subplots_adjust(wspace=0.3)

  fig.savefig(f"{figure_name}.pdf", bbox_inches='tight')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot figure 13')
  parser.add_argument('--log_dir', type=str, default='./backup_logs/', help='The directory of log files')
  args = parser.parse_args()
  print(f"{args=}")

  tflops, mem = extract(args.log_dir)
  plot([tflops, mem], seqlens, sys_names, figure_name='fig13', add_legend=True)