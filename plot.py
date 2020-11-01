# coding=utf-8
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot,savefig
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

# recipe = ["29 Android",
#           "26 IOS",
#           "9 Windows"]

# data = [float(x.split()[0]) for x in recipe]
# ingredients = [x.split()[-1] for x in recipe]

# def func(pct, allvals,ingredients):
#     absolute = int(pct/100.*np.sum(allvals))
#     return "{:.1f}%\n({:d}{:s})".format(pct, absolute)


# wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
#                                   textprops=dict(color="w"))

# ax.legend(wedges, ingredients,
#           title="Clients",
#           loc="center left",
#           bbox_to_anchor=(1, 0, 0.5, 1))

# plt.setp(autotexts, size=8, weight="bold")

# ax.set_title("The num of clients")
# import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Android', 'Windows', 'IOS'
# sizes = [29, 26, 9]
# # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=False, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()

# plt.savefig("client11.pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

name = ["Total Length", "TTL", "Option", "Host Name", "Vendor Class Identifier", "Parameter Request"]
Android = np.around(np.array([18.0, 1.0, 10.0, 24.0, 16.0, 8.0]) / 29. *100, 2)
iOS = np.around(np.array([1.0, 1.0, 2.0, 23.0, 1.0, 4.0]) / 26. *100, 2)
windows = np.around(np.array([5.0, 1.0, 4.0, 9.0, 2.0, 4.0]) / 9. *100, 2)
total = np.around((np.array([18.0, 1.0, 10.0, 24.0, 16.0, 8.0]) + np.array([1.0, 1.0, 2.0, 23.0, 1.0, 4.0]) \
                + np.array([5.0, 1.0, 4.0, 9.0, 2.0, 4.0])) / (29. + 26. + 9.) *100, 2)

x = np.arange(len(["Android", "IOS", "Windows", "Total"]))
total_width, n = 0.8, 6    # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2
width = 0.25

ip = np.array([Android[0], iOS[0], windows[0], total[0]])
print(ip)
ttl = np.array([Android[1], iOS[1], windows[1], total[1]])
option = np.array([Android[2], iOS[2], windows[2], total[2]])
host_name = np.array([Android[3], iOS[3], windows[3], total[3]])
vendor_class = np.array([Android[4], iOS[4], windows[4], total[4]])
Parameter_request = np.array([Android[5], iOS[5], windows[5], total[5]])

data = {"客户端类型": ["Android", "IOS", "Windows", "Total"], name[0]: ip, \
        name[1]: ttl, name[2]: option, name[3]: host_name, name[4]: vendor_class, \
        name[5]:Parameter_request}
df = pd.DataFrame(data)
df.plot(x = "客户端类型", y = name, kind = "bar", figsize = (16, 6), width = 0.7, rot = 0)

# plt.bar(x, ip,  width=width, label=name[0],color='darkorange')
# plt.bar(x + width, ttl, width=width, label=name[1], color='deepskyblue')
# plt.bar(x + 2 * width, option, width=width, label=name[2], color='green', tick_label=["Android", "IOS", "Windows", "Total"])
# plt.bar(x + 3 * width, host_name, width=width, label=name[3], color='red')
# plt.bar(x + 4 * width, vendor_class, width=width, label=name[4])
# plt.bar(x + 5 * width, Parameter_request, width=width, label=name[5])


# # 显示在图形上的值
# for a, b in zip(x,ip):
#     plt.text(a, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x,ttl):
#     plt.text(a+width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, option):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, host_name):
#     plt.text(a+3*width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, vendor_class):
#     plt.text(a+4*width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, Parameter_request):
#     plt.text(a+5*width, b+0.1, b, ha='center', va='bottom')

the_fontsize = 18
plt.xticks(fontsize=the_fontsize)
plt.yticks(fontsize=the_fontsize)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=14)  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('特征独特比（%）', fontsize=the_fontsize)
plt.xlabel('终端类型', fontsize=the_fontsize)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 尺寸
plt.title("特征独特性", fontsize=the_fontsize)
plt.tight_layout()
plt.savefig('uniqueness.png')
plt.close()
# plt.show()


# #######################################################################################################
# 
# #######################################################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

the_fontsize = 10

labels = ["Android", "IOS", "Windows", "Total"]
dhcp_ua_6 = [62.0, 98.5, 71.9, 77.4]
dhcp_ua_6_1 = [92.0, 99.7, 95.2, 95.4]

dhcp_ua = [60.8, 15.3, 0, 34.5]
dhcp_ua_1 = [89.5, 25.4, 25.8, 55.8]

dhcp = [0., 0., 0., 0.]
dhcp_1 = [64.3, 9.0, 25.8, 37.6]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dhcp_ua_6, width, label='侧重精准率')
rects2 = ax.bar(x + width/2, dhcp_ua_6_1, width, label='侧重召回率')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('终端类型', fontsize=the_fontsize)
ax.set_ylabel('准确率 (%)', fontsize=the_fontsize)
ax.set_title('DHCP+UA+DUID算法结果', fontsize=the_fontsize)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=the_fontsize)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=the_fontsize)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.tight_layout()
plt.savefig("dhcp_ua_6.png")

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dhcp_ua, width, label='侧重精准率')
rects2 = ax.bar(x + width/2, dhcp_ua_1, width, label='侧重召回率')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('终端类型', fontsize=the_fontsize)
ax.set_ylabel('准确率 (%)', fontsize=the_fontsize)
ax.set_title('DHCP+UA算法结果', fontsize=the_fontsize)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=the_fontsize)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=the_fontsize)
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.tight_layout()
plt.savefig("dhcp_ua.png")

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dhcp, width, label='侧重精准率')
rects2 = ax.bar(x + width/2, dhcp_1, width, label='侧重召回率')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('终端类型', fontsize=the_fontsize)
ax.set_ylabel('准确率 (%)', fontsize=the_fontsize)
ax.set_title('DHCP算法结果', fontsize=the_fontsize)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=the_fontsize)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=the_fontsize)
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.tight_layout()
plt.savefig("dhcp.png")
plt.close()


# #######################################################################################################
# 
# #######################################################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

the_fontsize = 14
title_list = [
"Android-侧重精准率",
"Android-侧重召回率",
"IOS-侧重精准率",
"IOS-侧重召回率",
"Windows-侧重精准率",
"Windows-侧重召回率",
"Total-侧重精准率",
"Total-侧重召回率"
]
data_list = [
np.array([0., 60.8, 62.0]),
np.array([64.3, 89.5, 92.0]),
np.array([0., 15.3, 98.5]),
np.array([9.0, 25.4, 99.7]),
np.array([0., 0., 71.9]),
np.array([25.8, 25.8, 95.2]),
np.array([0., 34.5, 77.4]),
np.array([37.6, 55.8, 95.4])
]
x_list = ["DHCP", "DHCP+UA", "DHCP+UA+DUID"]
marker_type = ["^", "s", "*",  "D", "p", "o", "+", "h"]
plt.figure(figsize=(8,4))
for line_i in range(len(title_list)):
    plt.plot(x_list, data_list[line_i], marker=marker_type[line_i], label=title_list[line_i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=the_fontsize)
plt.xticks(fontsize=the_fontsize)
plt.yticks(fontsize=the_fontsize)
plt.xlabel("算法框架", fontsize=the_fontsize)
plt.ylabel("识别准确率", fontsize=the_fontsize)
plt.tight_layout()
plt.savefig("total.png", fontsize=the_fontsize)
plt.close()
