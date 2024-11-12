import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class FloodSegmentation:
    """
    FloodSegmentation 类用于处理洪水分割数据。
    属性:
        flow_file_paths (str): 流量数据 CSV 文件的路径。
        rain_file_paths (str): 降雨数据 CSV 文件的路径。
        column_num (int): 用于提取目标和降雨数据的列号。
        theta (int): 用于过滤的阈值。
        df (DataFrame): 包含流量数据的 DataFrame。
        targets (ndarray): 从流量数据中提取的目标值数组。
        rainfall_df (DataFrame): 包含降雨数据的 DataFrame。
        rainfall (ndarray): 从降雨数据中提取的降雨值数组。
        times (DatetimeIndex): 降雨数据的日期时间索引。
        tar (list): 用于存储过滤后的目标值的列表。
        index (list): 用于存储超过某个阈值的目标索引的列表。
        groups (list): 用于存储索引组的列表。
    方法:
        filter_index(index):
            过滤并分组连续的索引。
        prepare_targets():
            准备目标值并识别用于处理的索引。
        nb(groups):
            处理索引组以识别显著的段落。
        inner_process(news):
            进一步处理识别的段落以细化结果。
        rainfall_filter(rainfall, news):
            根据识别的段落过滤降雨数据。
        process():
            主处理方法，协调工作流程。
        plot():
            可视化流量和降雨数据以及识别的段落。
    """

    def __init__(self, config, column_num):
        self.flow_file_paths = config.get("path_out", "new.csv")
        self.rain_file_paths = config.get("rain_file_paths", "basin_rainfall.csv")
        self.column_num = column_num
        self.theta = config.get("theta", 24)

        self.df = pd.read_csv(self.flow_file_paths, encoding="utf-8-sig", encoding_errors='ignore')
        self.targets = self.df.iloc[:, self.column_num + 1].values

        self.rainfall_df = pd.read_csv(self.rain_file_paths, encoding="utf-8-sig", encoding_errors='ignore')
        self.rainfall = self.rainfall_df.iloc[:, self.column_num].values

        self.times = pd.to_datetime(self.rainfall_df.iloc[:, 0])

        self.tar = []
        self.index = []
        self.groups = []

    def filter_index(self, index):
        index = np.sort(index)
        groups = []
        group_result = []
        for i in index:
            if groups and i - 1 == groups[-1][-1]:
                groups[-1].append(i)
            else:
                groups.append([i])
        for i, group in enumerate(groups):
            new_group = [group[0], group[-1]]
            group_result.append(new_group)
        return group_result

    def prepare_targets(self):
        for i in range(len(self.targets)):
            if self.targets[i] > 0:
                self.tar.append(self.targets[i])
        self.tar = np.sort(self.tar)
        b = int(0.60 * len(self.tar))
        for i in range(len(self.targets)):
            if self.targets[i] > self.tar[b]:
                self.index.append(i)
        self.groups = self.filter_index(self.index)

    def nb(self, groups):
        nums = []
        for n in range(999):
            new = []
            for i in range(len(groups)):
                if i == len(groups) - 1:
                    new.append(groups[i])
                    continue
                if groups[i][1] >= groups[i + 1][0]:
                    groups[i][1] = max(groups[i][1], groups[i + 1][1])
                elif groups[i][1] - groups[i][0] < 5:
                    continue
                else:
                    item = self.targets[groups[i][0]:groups[i][1] + 1]
                    max_item = max(item)
                    max_item_index = np.argmax(item) + groups[i][0]
                    min_item = min(item)
                    item_fu = self.targets[groups[i + 1][0]:groups[i + 1][1] + 1]
                    max_fu = max(item_fu)
                    max_item_fu = np.argmax(item_fu) + groups[i + 1][0]
                    if max_item >= self.tar[int(0.99 * len(self.tar))]:
                        new.append(groups[i])
                    else:
                        if min_item <= 3 / 4 * max_item and abs(max_item_fu - max_item_index) >= self.theta:
                            new.append(groups[i])
                        else:
                            continue
            nums.append(len(new))
            groups = new
            if n >= 2 and nums[n] == nums[n - 1]:
                break
        unique_new = []
        seen = set()
        for segment in new:
            if tuple(segment) not in seen:
                unique_new.append(segment)
                seen.add(tuple(segment))
        return unique_new

    def inner_process(self, news):
        result = []
        index = []
        for i in range(len(news)):
            start = news[i][0]
            end = news[i][1]
            tar = self.targets[start:end]
            sort = np.sort(tar)
            b = int(0.4 * len(sort))
            for j in range(len(tar)):
                if tar[j] > sort[b]:
                    index.append(j + start)
            result = self.filter_index(index)
        return result

    def rainfall_filter(self, rainfall, news):
        result = []
        for i in range(len(news)):
            start = news[i][0]
            zero_count = 0
            while start > 0 and zero_count < 3:
                if rainfall[start] == 0:
                    zero_count += 1
                else:
                    zero_count = 0
                start -= 1
            end = news[i][1] + 1
            while end < len(self.targets) - 2 and (self.targets[end] > self.targets[end + 1] or self.targets[end+1] > self.targets[end + 2]):
                end += 1
            result.append([start, end])
        return result

    def process(self):
        self.prepare_targets()
        new = self.nb(self.groups)
        pp = self.inner_process(new)
        pp = self.nb(pp)
        result = self.rainfall_filter(self.rainfall, pp)
        return result
    
    def plot(self):
        result = np.array(self.groups)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.times, y=self.targets, mode='lines', name='流量 (m3/s)', line=dict(color='blue', width=2)))
        fig.add_trace(go.Bar(x=self.times, y=self.rainfall, name='降雨量 (mm)',
                            marker=dict(color='olive', line=dict(width=2, color='olive')), yaxis='y2'))
        yaxis = 'y2'
        for interval in result:
            if interval[1] < len(self.times):
                fig.add_vrect(x0=self.times[interval[0]], x1=self.times[interval[1]], fillcolor='peachpuff', opacity=0.3, line_width=0)
        fig.update_layout(
            title=f'场次划分结果',
            title_x=0.5,
            xaxis_title='时间',
            yaxis_title='流量 (m3/s)',
            title_font=dict(size=40),
            xaxis=dict(
                title_font=dict(size=25),
                tickfont=dict(size=15)
            ),
            yaxis=dict(
                title_font=dict(size=25),
                tickfont=dict(size=15),
                range=[0, max(self.targets) * 2]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_layout(
            yaxis2=dict(
                title='降雨量 (mm)',
                overlaying='y',
                side='right',
                range=[0, max(self.rainfall) * 1.5],
                autorange='reversed',
                title_font=dict(size=25),
                tickfont=dict(size=15)
            )
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()

if __name__ == "__main__":
    config = {
        "flow_file_paths": "new.csv",  #实际流量数据
        "rain_file_paths": "basin_rainfall.csv",  #降雨数据
        "column_num": 1,  #流域第几列
        "theta": 24  #默认
    }

    flood_segmentation = FloodSegmentation(config)
    result = flood_segmentation.process()
    flood_segmentation.plot()
    # print(result)