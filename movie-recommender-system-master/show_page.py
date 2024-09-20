import tkinter as tk
from views import *
class MainPage:
    def __init__(self, master: tk.Tk, uid):
        self.root = master
        self.root.geometry("400x670")
        self.root.title('电影推荐系统')
        self.uid = uid
        self.create_page()

    def create_page(self):
        self.history_frame = HistoryFrame(self.root, self.uid)
        self.recommend_frame = RecommendFrame(self.root, self.uid)

        menubar = tk.Menu(self.root)
        menubar.add_command(label='历史', command=self.show_history)
        menubar.add_command(label='推荐', command=self.show_recommend)
        self.root['menu'] = menubar
    # 点击事件
    def show_history(self):
        self.history_frame.pack()
        self.recommend_frame.pack_forget()

    def show_recommend(self):
        self.recommend_frame.pack()
        self.history_frame.pack_forget()


if __name__ == '__main__':
    root = tk.Tk()
    MainPage(master=root)
    root.mainloop()