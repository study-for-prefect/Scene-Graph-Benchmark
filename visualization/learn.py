import json

# 仅加载必需文件
PRED_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_prediction.json'
INFO_PATH = r'D:\school\pythonProject\SSG\Scene-Graph-Benchmark\output_dir\custom_data_info.json'

with open(PRED_PATH, 'r') as f:
    pred_data = json.load(f)['0']   # 选择图片索引
with open(INFO_PATH, 'r') as f:
    info_data = json.load(f)    # 加载数据

classes = info_data['ind_to_classes']   # ind_to_classes是类别索引
predicates = info_data['ind_to_predicates'] # ind_to_predicates是关系索引

# 提取框标签列表
box_labels = [classes[idx] for idx in pred_data['bbox_labels']] #

# 任务：在此处编写循环，遍历 pred_data['rel_pairs']

# 取前 10 个关系，输出格式： "Person => riding => Bike"


















from PIL import Image,ImageDraw
# 创建一张500*500 的黑色侧视图
img = Image.new('RGB',(500,500),'BLACK')
draw = ImageDraw.Draw(img)
# 任务：查看文档，在坐标(50, 50)到(250, 250)绘制一个线宽为3的绿色矩形
# # 任务：在中心点(150, 150)写入白色文字 "Test"
draw.rectangle(((50,50),(250,250)),outline='Green',width=3)
draw.text((150,150),'TEST',fill='white')
img.show()



import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
# 仅添加三个节点和两条边
G.add_edge("Person_0", "Cup_1", label="holding")
G.add_edge("Person_0", "Chair_2", label="sitting on")

# 任务：查阅 nx.draw 文档，将节点画成红色，边画成蓝色，并显示标签
pos = nx.spring_layout(G)
nx.draw(G,pos,node_color='red',edge_color='blue',with_labels=True,node_size=2000,arrows=True)
# 在此处补全绘图代码
nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'label'),font_color='red')
plt.show()