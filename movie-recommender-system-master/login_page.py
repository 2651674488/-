import tkinter as tk
from tkinter import messagebox
import controller as ct
from show_page import *
from new_user_page import *


class LoginPage:

    def __init__(self, master):
        self.root = master
        self.root.geometry("400x200")
        self.root.title("登录")
        # 获取用户id
        self.userId = tk.StringVar()
        self.page = tk.Frame(root)
        self.page.pack()

        tk.Label(self.page).grid(row=0, column=0)
        tk.Label(self.page, text="用户id:").grid(row=1, column=1)
        tk.Entry(self.page, textvariable=self.userId).grid(row=1, column=2)
        tk.Button(self.page, text="登录", command=self.login).grid(
            row=2, column=1, pady=20
        )
        tk.Button(self.page, text="新用户", command=self.new_user).grid(row=2, column=2)
        tk.Button(self.page, text="更新推荐", command=self.update_recommend).grid(
            row=2, column=3
        )
        root.mainloop()

    def update_recommend(self):
        ap.login_initialisation()
        messagebox.showwarning(title="提示", message="更新完成，欢迎使用~")

    def login(self):
        uid = self.userId.get()
        # 检查用户id是否存在
        flag = ct.check_userid(int(uid))
        if flag:
            print("登陆成功")
            self.page.destroy()
            ap.login_initialisation()
            MainPage(self.root, uid)
        else:
            print("登陆失败")
            messagebox.showwarning(
                title="警告", message="请正确输入用户id，或直接点击新用户"
            )

    def new_user(self):
        new_userId = ct.register()
        print("新用户注册")
        self.page.destroy()
        NewUserPage(self.root, new_userId)


if __name__ == "__main__":
    root = tk.Tk()
    LoginPage(master=root)
