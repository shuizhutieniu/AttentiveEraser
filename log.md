

在DIffusion模型中，通过编辑其中的Attention map，我成功实现了在不改变提示语的情况下，去除图片中的某个物体，并且修改后的图片的结构几乎不发生改变。
具体来说，我是通过目标词的attention map获取一个mask，然后应用在不同的层、不同的词上。
请你为这个项目写一个运行脚本`run.py`
实现在终端输入参数后运行

接收的参数如下：
`p`，即`prompt`，字符串类型， 例如`"A squirrel and a cherry"`
`s`，即`seed`，int型，例如`42`
`i`，即`index`，int型，即目标词的位置，例如在`"A squirrel and a cherry"`中删除`cherry`，输入5，因为5代表第五个单词`cherry`
`t`，即`step`，列表型，DIffusion一共有50层，例如输入`[1, 20]`代表删除1-20层
`w`，即`word`，列表型，DIffusion的attention map一共有77层，代表最多能接受77个词，这个参数代表要用mask对哪些词的attention map进行操作，默认值是`i`，即只改变目标词的attention map

请注意：你只需要写出这个脚本的接收参数部分，接收参数后的调用模型部分我已经写好。
请注意：你应该用英文写参数的解释，并且应该加`-h`参数来获取帮助。另外，我给出的参数解释可能有一些不规范，请你在不修改含义的基础上调整得更专业一些。



第一次使用要一段时间加载模型



`python run.py -p "A squirrel and a cherry" -s 2436247 -i 5 -t 1 20`


`Gaussian Blur`



在DIffusion模型中，通过编辑其中的Attention map，我成功实现了在不改变提示语的情况下，去除图片中的某个物体，并且修改后的图片的结构几乎不发生改变。
具体来说，我是通过目标词的attention map获取一个mask，然后应用在不同的层、不同的词上。


请你为这个项目的github仓库写一份readme.md（英文）

请你参考以下内容（请注意：我的表述可能不专业，请你在我的基础上修改得专业一些）：
The codebase is tested under NVIDIA Tesla T4 with the python library pytorch-2.2.1+cu121 and diffusers-0.16.1
我们强烈建议使用特定版本的Diffusers，因为它正在不断更新

先使用pip安装仓库中的requirements.txt

然后请你查看`run.py`中各个参数的含义，然后写在readme.md中

接着举一个例子，例如使用命令`python run.py -p "A squirrel and a cherry" -s 2436247 -i 5 -t 1 20`来执行，或者请读者使用`python run.py -h`来查看