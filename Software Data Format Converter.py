################# Grid_Optimization_Results.xlsx
# // 帕累托数据 - 第二轮优化 [经济性, 碳排放(吨CO2e), 方案ID]
# const paretoData = ref([
#   [0.82, 250, 1],
#   [0.77, 180, 2],
#   [0.88, 300, 3],
#   [0.74, 160, 4],
#   [0.92, 360, 5],
#   [0.85, 270, 6],
#   [0.79, 200, 7],
#   [0.90, 320, 8],
#   [0.76, 190, 9],
#   [0.87, 280, 10],
#   [0.83, 240, 11],
#   [0.81, 220, 12],
#   [0.86, 260, 13],
#   [0.78, 210, 14],
#   [0.89, 310, 15]
# ])
###################
import pandas as pd

# 读取上传的 Excel 文件
file_path = "C:\Users\86183\Desktop\111\Grid_Optimization_Results.xlsx"
df = pd.read_excel(file_path)

# 检查列名
columns = df.columns.tolist()

# 假设年化总成本在第1列，碳排放量在第2列（单位为 kg），我们转换为吨（t）
# 并归一化成本为 [0, 1] 之间的小数（比如用最大值归一）
costs = df[columns[1]]
emissions_kg = df[columns[2]]
normalized_costs = (costs - costs.min()) / (costs.max() - costs.min())
emissions_ton = emissions_kg / 1000.0

# 构造目标数据结构
pareto_data = []
for i in range(len(df)):
    cost = round(normalized_costs.iloc[i], 2)
    emission = round(emissions_ton.iloc[i], 0)
    pareto_data.append([cost, int(emission), i + 1])

# 构造输出文本
lines = ["// 帕累托数据 - 第二轮优化 [经济性, 碳排放(吨CO2e), 方案ID]", "//改改改", "const paretoData = ref(["]
for item in pareto_data:
    lines.append(f"  {item},")
lines[-1] = lines[-1].rstrip(',')  # 移除最后一个逗号
lines.append("])")

# 合并为文本输出
output_text = "\n".join(lines)
output_text[:500]  # 只显示前500字符预览，如需保存全部请说明
