import tkinter as tk
from tkinter import messagebox
import controller as ct
import views
import Application as ap


class NewUserPage:
    def __init__(self, master: tk.Tk, new_userId):
        self.select_genres = None
        self.root = master
        self.root.geometry("400x650")
        self.root.title("电影推荐系统")
        self.userId = new_userId
        self.page = tk.Frame(self.root)
        self.page.pack()
        self.create_page()

    def create_page(self):
        tk.Label(self.page).pack()
        tk.Label(self.page, text=f"用户id:{self.userId}").pack()
        tk.Label(self.page, text="请选择你感兴趣的类型（可多选）:").pack()
        self.ck1 = tk.IntVar()
        self.ck2 = tk.IntVar()
        self.ck3 = tk.IntVar()
        self.ck4 = tk.IntVar()
        self.ck5 = tk.IntVar()
        self.ck6 = tk.IntVar()
        self.ck7 = tk.IntVar()
        self.ck8 = tk.IntVar()
        self.ck9 = tk.IntVar()
        self.ck10 = tk.IntVar()
        self.check_buttons = {}
        self.check_buttons["战争"] = tk.Checkbutton(
            self.page, text="战争", variable=self.ck1, onvalue=1, offvalue=0
        )
        self.check_buttons["惊悚"] = tk.Checkbutton(
            self.page, text="惊悚", variable=self.ck2, onvalue=1, offvalue=0
        )
        self.check_buttons["犯罪"] = tk.Checkbutton(
            self.page, text="犯罪", variable=self.ck3, onvalue=1, offvalue=0
        )
        self.check_buttons["动画"] = tk.Checkbutton(
            self.page, text="动画", variable=self.ck4, onvalue=1, offvalue=0
        )
        self.check_buttons["动作"] = tk.Checkbutton(
            self.page, text="动作", variable=self.ck5, onvalue=1, offvalue=0
        )
        self.check_buttons["剧情"] = tk.Checkbutton(
            self.page, text="剧情", variable=self.ck6, onvalue=1, offvalue=0
        )
        self.check_buttons["爱情"] = tk.Checkbutton(
            self.page, text="爱情", variable=self.ck7, onvalue=1, offvalue=0
        )
        self.check_buttons["冒险"] = tk.Checkbutton(
            self.page, text="冒险", variable=self.ck8, onvalue=1, offvalue=0
        )
        self.check_buttons["悬疑"] = tk.Checkbutton(
            self.page, text="悬疑", variable=self.ck9, onvalue=1, offvalue=0
        )
        self.check_buttons["科幻"] = tk.Checkbutton(
            self.page, text="科幻", variable=self.ck10, onvalue=1, offvalue=0
        )
        for check in self.check_buttons.values():
            check.pack(anchor="w")
        tk.Button(self.page, text="确定", command=self.print_selected_genres).pack()

    def print_selected_genres(self):
        self.select_genres = []
        if self.ck1.get() == 1:
            self.select_genres.append("战争")
        if self.ck2.get() == 1:
            self.select_genres.append("惊悚")
        if self.ck3.get() == 1:
            self.select_genres.append("犯罪")
        if self.ck4.get() == 1:
            self.select_genres.append("动画")
        if self.ck5.get() == 1:
            self.select_genres.append("动作")
        if self.ck6.get() == 1:
            self.select_genres.append("剧情")
        if self.ck7.get() == 1:
            self.select_genres.append("爱情")
        if self.ck8.get() == 1:
            self.select_genres.append("冒险")
        if self.ck9.get() == 1:
            self.select_genres.append("悬疑")
        if self.ck10.get() == 1:
            self.select_genres.append("科幻")
        if len(self.select_genres) == 0:
            messagebox.showwarning(title="警告", message="请至少选择一种类型")
        else:
            # todo 用新人推荐=======================================
            self.page.destroy()
            NewUserRecommend(self.root, self.userId, self.select_genres).pack()


class NewUserRecommend(tk.Frame):
    def __init__(self, root, uid, select_genres):
        super().__init__(root)
        self.uid = uid
        self.times = 0
        self.movie_dict = {}
        self.score = tk.StringVar()
        self.select_genres = select_genres
        self.create_page()

    def create_page(self):
        tk.Label(self, text="推荐页面").pack()
        # movieId	name	alias	actors	directors	doubanScore	genres	languages	mins	regions	storyline	tags	year
        self.movieId, self.movie_dict = ct.new_user_recommend_movie(
            self.select_genres, self.uid, self.times
        )
        self.label1 = tk.Label(self, text=f"刷新次数{self.times}")
        self.user_id_label = tk.Label(self, text=f"用户id:{self.uid}")
        self.user_id_label.pack(anchor="w")
        # self.movie_id_label = tk.Label(self, text=f'电影ID:{self.movie_dict.get("movieId", "unknown")}')
        # self.movie_id_label.pack(anchor='w')
        self.movie_name_label = tk.Label(
            self, text=f'电影:{self.movie_dict.get("name", "unknown")}'
        )
        self.movie_name_label.pack(anchor="w")
        self.alias_label = tk.Label(
            self, text=f'电影别名:{self.movie_dict.get("alias", "unknown")}'
        )
        self.alias_label.pack(anchor="w")
        tk.Label(self, text=f"演员:").pack(anchor="w")
        # text
        self.actor_text = tk.Text(self, height=5, width=65, wrap=tk.WORD)
        self.actor_text.pack()
        # 如果更新，先删除后插入
        self.actor_text.insert("end", self.movie_dict.get("actors", "unknown"))
        self.director_label = tk.Label(
            self, text=f'导演:{self.movie_dict.get("directors", "unknown")}'
        )
        self.director_label.pack(anchor="w")
        self.doubanScore_label = tk.Label(
            self, text=f'豆瓣评分:{self.movie_dict.get("doubanScore", "unknown")}'
        )
        self.doubanScore_label.pack(anchor="w")
        self.genres_label = tk.Label(
            self, text=f'类型:{self.movie_dict.get("genres", "unknown")}'
        )
        self.genres_label.pack(anchor="w")
        self.languages_label = tk.Label(
            self, text=f'语言:{self.movie_dict.get("languages", "unknown")}'
        )
        self.languages_label.pack(anchor="w")
        self.mins_label = tk.Label(
            self, text=f'时长:{self.movie_dict.get("mins", "unknown")}'
        )
        self.mins_label.pack(anchor="w")
        self.regions_label = tk.Label(
            self, text=f'地区:{self.movie_dict.get("regions", "unknown")}'
        )
        self.regions_label.pack(anchor="w")
        tk.Label(self, text=f"简介:").pack(anchor="w")
        # text
        self.storyline_text = tk.Text(self, height=10, width=65, wrap=tk.WORD)
        self.storyline_text.pack()
        # 如果更新，先删除后插入
        self.storyline_text.insert("end", self.movie_dict.get("storyline", "unknown"))
        self.tags_label = tk.Label(
            self, text=f'标签:{self.movie_dict.get("tags", "unknown")}'
        )
        self.tags_label.pack(anchor="w")
        self.year_label = tk.Label(
            self, text=f'上映时间:{self.movie_dict.get("year", "unknown")}'
        )
        self.year_label.pack(anchor="w")
        # 打分
        tk.Label(self, text="请输入电影评分1~5分:").pack(anchor="w")
        self.score_entry = tk.Entry(self, textvariable=self.score)
        self.score_entry.pack(anchor="w")
        # 确定分数
        tk.Button(self, text="确定", command=self.get_score).pack(anchor="w")
        # more
        tk.Button(self, text="next", command=self.next_movie).pack(anchor="e")
        # ========显示标签========
        self.label1.pack(anchor="w")

    def get_score(self):
        score = self.score.get()
        try:
            # 尝试将字符串转换为整数
            score = int(score)
            # 检查分数是否在1到5之间
            if 1 <= score <= 5:
                ct.grade_movie(self.uid, self.movieId, score)
                messagebox.showwarning(title="提示", message="添加成功")
            else:
                print("失败")
        except ValueError:
            # 如果转换失败，说明输入不是整数
            messagebox.showwarning(title="警告", message="请输入1~5之间的评分")

    def next_movie(self):
        self.times += 1
        self.label1.config(text=f"刷新次数{self.times}")
        # 获得电影信息
        self.movieId, self.movie_dict = ct.new_user_recommend_movie(
            self.select_genres, self.uid, self.times
        )
        # =========更新内容========
        self.movie_name_label.config(
            text=f'电影:{self.movie_dict.get("name", "unknown")}'
        )
        self.alias_label.config(
            text=f'电影别名:{self.movie_dict.get("alias", "unknown")}'
        )
        # '1.0'是Tkinter中指定位置的格式，其中1表示第一行，0表示该行的第一个字符,'end'是一个特殊的索引，它表示文本的最后一个字符。
        self.actor_text.delete("1.0", "end")
        self.actor_text.insert("end", self.movie_dict.get("actors", "unknown"))
        self.director_label.config(
            text=f'导演:{self.movie_dict.get("directors", "unknown")}'
        )
        self.doubanScore_label.config(
            text=f'豆瓣评分:{self.movie_dict.get("doubanScore", "unknown")}'
        )
        self.genres_label.config(
            text=f'类型:{self.movie_dict.get("genres", "unknown")}'
        )
        self.languages_label.config(
            text=f'语言:{self.movie_dict.get("languages", "unknown")}'
        )
        self.mins_label.config(text=f'时长:{self.movie_dict.get("mins", "unknown")}')
        self.regions_label.config(
            text=f'地区:{self.movie_dict.get("regions", "unknown")}'
        )
        self.storyline_text.delete("1.0", "end")
        self.storyline_text.insert("end", self.movie_dict.get("storyline", "unknown"))
        self.tags_label.config(text=f'标签:{self.movie_dict.get("tags", "unknown")}')
        self.year_label.config(
            text=f'上映时间:{self.movie_dict.get("year", "unknown")}'
        )
        self.score_entry.delete(0, "end")


if __name__ == "__main__":
    root = tk.Tk()
    # todo
    new_userId = ct.register()
    NewUserPage(master=root, new_userId=44)
    root.mainloop()
