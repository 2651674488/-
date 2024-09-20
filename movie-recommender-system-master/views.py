import tkinter as tk
import tkinter.ttk
import controller as ct
from tkinter import messagebox
import Application as ap

# 继承Frame对象
class HistoryFrame(tk.Frame):
    def __init__(self, root, uid):
        # root参数被传递给父类的构造函数，这样HistoryFrame就可以作为一个框架
        super().__init__(root)
        self.uid = uid
        self.create_page()

    def create_page(self):
        tk.Label(self, text=f'用户id:{self.uid}').pack()
        columns = ("name", "rating")
        # columns_values = ("电影", "评分")
        self.tree_view = tkinter.ttk.Treeview(self, show='headings', columns=columns)
        self.tree_view.column('name', width=300, anchor='center')
        self.tree_view.column('rating', width=40, anchor='center')
        self.tree_view.heading('name', text='电影')
        self.tree_view.heading('rating', text='评分')
        self.tree_view.pack(fill=tk.BOTH, expand=True)
        self.show_data()
        tk.Button(self, text='刷新数据', command=self.show_data).pack(anchor=tk.E, pady=5)
    def show_data(self):
        # 删除旧的数据，避免重复
        for _ in map(self.tree_view.delete, self.tree_view.get_children('')):
            pass
        movie_dict = ct.rating_history(self.uid)
        index = 0
        if movie_dict is None:
            # 无历史记录
            pass
        else:
            for m_key, m_value in movie_dict.items():
                # 树，父节点填空
                self.tree_view.insert('', index + 1, values=(
                    m_key, m_value
                ))

class RecommendFrame(tk.Frame):
    def __init__(self, root, uid):
        super().__init__(root)
        self.uid = uid
        self.times = 0
        self.movie_dict = {}
        self.score = tk.StringVar()
        self.create_page()
    def create_page(self):
        tk.Label(self, text='推荐页面').pack()
        # movieId	name	alias	actors	directors	doubanScore	genres	languages	mins	regions	storyline	tags	year
        self.movieId, self.movie_dict = ct.recommend_movie(self.uid, self.times)
        self.label1 = tk.Label(self, text=f'刷新次数{self.times}')
        self.user_id_label = tk.Label(self, text=f'用户id:{self.uid}')
        self.user_id_label.pack(anchor='w')
        # self.movie_id_label = tk.Label(self, text=f'电影ID:{self.movie_dict.get("movieId", "unknown")}')
        # self.movie_id_label.pack(anchor='w')
        self.movie_name_label = tk.Label(self, text=f'电影:{self.movie_dict.get("name", "unknown")}')
        self.movie_name_label.pack(anchor='w')
        self.alias_label = tk.Label(self, text=f'电影别名:{self.movie_dict.get("alias", "unknown")}')
        self.alias_label.pack(anchor='w')
        tk.Label(self, text=f'演员:').pack(anchor='w')
        # text
        self.actor_text = tk.Text(self, height=5, width=65, wrap=tk.WORD)
        self.actor_text.pack()
        # 如果更新，先删除后插入
        self.actor_text.insert('end', self.movie_dict.get("actors", "unknown"))
        self.director_label = tk.Label(self, text=f'导演:{self.movie_dict.get("directors", "unknown")}')
        self.director_label.pack(anchor='w')
        self.doubanScore_label = tk.Label(self, text=f'豆瓣评分:{self.movie_dict.get("doubanScore", "unknown")}')
        self.doubanScore_label.pack(anchor='w')
        self.genres_label = tk.Label(self, text=f'类型:{self.movie_dict.get("genres", "unknown")}')
        self.genres_label.pack(anchor='w')
        self.languages_label = tk.Label(self, text=f'语言:{self.movie_dict.get("languages", "unknown")}')
        self.languages_label.pack(anchor='w')
        self.mins_label = tk.Label(self, text=f'时长:{self.movie_dict.get("mins", "unknown")}')
        self.mins_label.pack(anchor='w')
        self.regions_label = tk.Label(self, text=f'地区:{self.movie_dict.get("regions", "unknown")}')
        self.regions_label.pack(anchor='w')
        tk.Label(self, text=f'简介:').pack(anchor='w')
        # text
        self.storyline_text = tk.Text(self, height=10, width=65, wrap=tk.WORD)
        self.storyline_text.pack()
        # 如果更新，先删除后插入
        self.storyline_text.insert('end', self.movie_dict.get("storyline", "unknown"))
        self.tags_label = tk.Label(self, text=f'标签:{self.movie_dict.get("tags", "unknown")}')
        self.tags_label.pack(anchor='w')
        self.year_label = tk.Label(self, text=f'上映时间:{self.movie_dict.get("year", "unknown")}')
        self.year_label.pack(anchor='w')
        # 打分
        tk.Label(self, text='请输入电影评分1~5分:').pack(anchor='w')
        self.score_entry = tk.Entry(self, textvariable=self.score)
        self.score_entry.pack(anchor='w')
        # 确定分数
        tk.Button(self, text="确定", command=self.get_score).pack(anchor='w')
        # more
        tk.Button(self, text="next", command=self.next_movie).pack(anchor='e')
        # ========显示标签========
        self.label1.pack(anchor='w')


    def get_score(self):
        score = self.score.get()
        try:
            # 尝试将字符串转换为整数
            score = int(score)
            # 检查分数是否在1到5之间
            if 1 <= score <= 5:
                ct.grade_movie(self.uid, self.movieId, score)
                messagebox.showwarning(title='提示', message='添加成功')
            else:
                messagebox.showwarning(title='警告', message='请输入1~5之间的整数评分')
        except ValueError:
            # 如果转换失败，说明输入不是整数
            messagebox.showwarning(title='警告', message='请输入1~5之间的整数评分')

    def next_movie(self):
        self.times += 1
        self.label1.config(text=f'刷新次数{self.times}')
        # 获得电影信息
        self.movieId, self.movie_dict = ct.recommend_movie(self.uid, self.times)
        # =========更新内容========
        self.movie_name_label.config(text=f'电影:{self.movie_dict.get("name", "unknown")}')
        self.alias_label.config(text=f'电影别名:{self.movie_dict.get("alias", "unknown")}')
        # '1.0'是Tkinter中指定位置的格式，其中1表示第一行，0表示该行的第一个字符,'end'是一个特殊的索引，它表示文本的最后一个字符。
        self.actor_text.delete('1.0', 'end')
        self.actor_text.insert('end', self.movie_dict.get("actors", "unknown"))
        self.director_label.config(text=f'导演:{self.movie_dict.get("directors", "unknown")}')
        self.doubanScore_label.config(text=f'豆瓣评分:{self.movie_dict.get("doubanScore", "unknown")}')
        self.genres_label.config(text=f'类型:{self.movie_dict.get("genres", "unknown")}')
        self.languages_label.config(text=f'语言:{self.movie_dict.get("languages", "unknown")}')
        self.mins_label.config(text=f'时长:{self.movie_dict.get("mins", "unknown")}')
        self.regions_label.config(text=f'地区:{self.movie_dict.get("regions", "unknown")}')
        self.storyline_text.delete('1.0', 'end')
        self.storyline_text.insert('end', self.movie_dict.get("storyline", "unknown"))
        self.tags_label.config(text=f'标签:{self.movie_dict.get("tags", "unknown")}')
        self.year_label.config(text=f'上映时间:{self.movie_dict.get("year", "unknown")}')
        self.score_entry.delete(0, 'end')


