

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


再添加几个，参数名由你来想：
1.是否替换自注意力map（self attention map），布尔类型，默认为False
2.替换自注意力图的`step`，列表型，例如输入`[1, 20]`代表替换1-20层，当前一项为False时不生效，默认为[1, 50]
3.替换embedding的`index`，列表型，例如在`"A squirrel and a cherry"`中替换`cherry`的`embedding`，输入[5]，因为5代表第五个单词`cherry`，默认值为[]，即不替换





第一次使用要一段时间加载模型



`python run.py -p "A squirrel and a cherry" -s 2436247 -i 5 -t 1 20`


`Gaussian Blur`


看看p2p 照着写



<table style="border-collapse: collapse;width: 100%;">

  <tr>
      <th colspan="2" style="text-align: center; vertical-align: middle; padding: 2px; border: 1px solid white;">和宝宝<del>吃如论</del>烤冷面</th>
  </tr>

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2308052246_IMG_2024.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2308052246_IMG_2025.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>
</table>