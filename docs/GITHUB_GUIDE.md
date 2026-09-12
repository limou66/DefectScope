# 发布到 GitHub

项目源码和可发布实验记录已经准备好。data/、runs/、.venv/、.cache/ 均被忽略，真实数据、模型权重和本机缓存不应上传。

本机 GitHub CLI 在交付检查时尚未登录，因此当前没有创建远程仓库，也没有上传代码。请在终端亲自登录，不要把访问令牌发到聊天中：

~~~powershell
cd D:\xiangmu\DefectScope
gh auth login
~~~

按界面选择 GitHub.com、HTTPS 和浏览器登录。设置本仓库的提交身份，填写自己的真实信息或 GitHub 提供的 noreply 邮箱：

~~~powershell
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub noreply 邮箱"
git add README.md LICENSE THIRD_PARTY_NOTICES.md pyproject.toml requirements*.txt .gitignore .github src scripts tests docs reports start_demo.bat
git diff --cached --stat
git commit -m "Build reproducible visual anomaly inspection project"
gh repo create DefectScope --public --source=. --remote=origin --push
~~~

如果账号已有同名仓库，先确认要推到哪个仓库，不要覆盖它。也可以将最后一步的仓库名换成自己喜欢的名称。

发布后查看 Actions 测试；本地测试通过不代表远程 CI 已经运行。README 的 reports 和 docs 链接均为仓库内相对路径。

## 简历链接

用最终真实仓库地址，不要写尚未创建的链接。建议先完成 docs/LEARNING_GUIDE.md 中至少一项自己的实质改进，再将项目写入简历。提交记录应反映真实完成的工作，不要补造日期或虚构多人团队。
