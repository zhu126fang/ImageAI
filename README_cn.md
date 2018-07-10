- 汉化说明<br>
本文采用机器翻译，供个人学习使用，内容准确性不作任何保证，谢谢。<br>

- 测试环境<br>
Window系统测试通过(Win8.1)<br>
Aliyun服务其测试通过（Ubuntu16.04）<br>

- 打包为wheel<br>
python setup.py bdist_wheel --universal<br>

- 汉化对象名称
文件"imageai/Detection/__init__.py" 行55 <br>
使用中文时图片上的文字会乱码，可能是matplotlib库的问题。<br>

# ImageAI <br>
一个python库，旨在使开发人员能够使用简单的几行代码构建具有自包含深度学习和计算机视觉功能的应用程序和系统。<br><br>
An <b>AI Commons</b> project <a href="https://commons.specpal.science" >https://commons.specpal.science </a>
由 [Moses Olafenwa](https://twitter.com/OlafenwaMoses) 和 [John Olafenwa](https://twitter.com/johnolafenwa)开发和维护, brothers, creators of [TorchFusion](https://github.com/johnolafenwa/TorchFusion)
and Authors of [Introduction to Deep Computer Vision](https://john.specpal.science/deepvision)<hr>

在简洁的基础上构建, <b>ImageAI</b> 支持最先进的机器学习算法列表，用于图像预测，自定义图像预测，对象检测，视频检测，视频对象跟踪和图像预测培训. <b>ImageAI</b> 目前支持使用4种不同的机器学习算法进行图像预测和训练 在ImageNet-1000数据集上接受过培训。 <b>ImageAI</b> 还支持使用在COCO数据集上训练的RetinaNet进行对象检测，视频检测和对象跟踪。 <br> 终于, <b>ImageAI</b> 将为更广泛的人提供支持 计算机视觉的更多专业方面包括但不限于图像 在特殊环境和特殊领域的认可。<br><br>

<b>New Release : ImageAI 2.0.1</b>
<br> What's new:<br>
- 添加自定义图像预测模型培训 using SqueezeNet, ResNet50, InceptionV3 and DenseNet121 <br>
- 使用自定义训练模型添加自定义图像预测 and generated model class json  <br>
- 预览版：添加视频对象检测和视频自定义对象检测（对象跟踪）  <br>
- 为所有图像预测和对象检测任务添加文件，numpy数组和流输入类型（仅用于视频检测的文件输入）<br>
- 添加文件和numpy数组输出类型，用于图像中的对象检测和自定义对象检测  <br>
- 引入4种速度模式('normal', 'normal', 'faster', 'fastest')进行图像预测，使预测时间在“最快”时减少50％，同时保持预测精度  <br>
- 为所有物体检测和视频物体检测任务引入5种速度模式('normal', 'normal', 'faster', 'fastest' and 'flash') ，使检测时间减少80％以上  <br>
with 'flash' 检测精度与'minimum_percentage_probability'保持平衡，并保持在低值 <br>
- 引入帧检测率，允许开发人员调整视频中的检测间隔，有利于实时/接近实时结果。 <br><br><br>

<h3><b><u>目录</u></b></h3>
<a href="#dependencies" >&#9635 依赖关系</a><br>
<a href="#installation" >&#9635 安装</a><br>
<a href="#prediction" >&#9635 图像预测</a><br>
<a href="#detection" >&#9635 对象检测</a><br>
<a href="#videodetection" >&#9635 视频对象检测和跟踪</a><br>
<a href="#customtraining" >&#9635 自定义模型培训</a><br>
<a href="#customprediction" >&#9635 自定义图像预测</a><br>
<a href="#sample" >&#9635 示例应用程序</a><br>
<a href="#recommendation" >&#9635 AI Practice Recommendations</a><br>
<a href="#contact" >&#9635 联系开发人员</a><br>
<a href="#ref" >&#9635 参考</a><br>
<br><br>

<div id="dependencies"></div>
<h3><b><u>依赖关系</u></b></h3>
To use <b>ImageAI</b> 在您的应用程序开发中，您必须安装以下内容 安装之前的依赖项 <b>ImageAI</b> : 

 <br> <br>
       <span><b>- Python 3.5.1 (and later versions) </b>      <a href="https://www.python.org/downloads/" style="text-decoration: none;" >Download</a> (Support for Python 2.7 coming soon) </span> <br>
       <span><b>- pip3 </b>              <a href="https://pypi.python.org/pypi/pip" style="text-decoration: none;" >Install</a></span> <br>
       <span><b>- Tensorflow 1.4.0 (and later versions)  </b>      <a href="https://www.tensorflow.org/install/install_windows" style="text-decoration: none;" > Install</a></span> or install via pip <pre> pip3 install --upgrade tensorflow </pre> 
       <span><b>- Numpy 1.13.1 (and later versions) </b>      <a href="https://www.scipy.org/install.html" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install numpy </pre> 
       <span><b>- SciPy 0.19.1 (and later versions) </b>      <a href="https://www.scipy.org/install.html" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install scipy </pre> 
       <span><b>- OpenCV  </b>        <a href="https://pypi.python.org/pypi/opencv-python" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install opencv-python </pre> 
       <span><b>- Pillow  </b>       <a href="https://pypi.org/project/Pillow/2.2.1/" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install pillow </pre> 
       <span><b>- Matplotlib  </b>       <a href="https://matplotlib.org/users/installing.html" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install matplotlib </pre> 
       <span><b>- h5py  </b>       <a href="http://docs.h5py.org/en/latest/build.html" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install h5py </pre> 
       <span><b>- Keras 2.x  </b>     <a href="https://keras.io/#installation" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install keras </pre> 

<div id="installation"></div>
 <h3><b><u>安装</u></b></h3>
      直接运行下面的安装命令: <br><br>
    <span>      <b>pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl </b></span> <br><br> <br>
    或先下载Python Wheel文件 <a href="https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl" ><b> imageai-2.0.1-py3-none-any.whl</b></a> 再运行下面的安装命令: <br><br><span><b>pip3 install C:\User\MyUser\Downloads\imageai-2.0.1-py3-none-any.whl</b></span> <br><br>

<div id="prediction"></div>
<h3><b><u>图像预测</u></b></h3>
<b>ImageAI</b> 提供4种不同的算法和模型类型来执行图像预测，在ImageNet-1000数据集上进行训练。
提供用于图像预测的4种算法包括<b> SqueezeNet </b>，<b> ResNet </b>，<b> InceptionV3 </b>和<b> DenseNet </b>。
您将在下面找到使用ResNet50模型的示例预测结果以及图像下方的"教程&文档"链接。
单击链接以查看完整的示例代码，说明，最佳实践指南和文档。
<p><img src="images/1.jpg" style="width: 400px; height: auto;" /> 
    <pre>convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352</pre>
</p>

<a href="imageai/Prediction/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> 教程&文档 </button></a><br><br>

<div id="detection"></div>
<h3><b><u>对象检测</u></b></h3>
<b>ImageAI</b> 对图像执行对象检测并从图像中提取每个对象提供了非常方便和强大的方法。 
提供的对象检测类仅支持目前最先进的RetinaNet，但可以选择调整最先进的性能或实时处理。
您将在下面找到使用RetinaNet模型的示例对象检测结果，单击"教程和文档"链接以查看完整的示例代码。
    <div style="width: 600px;" >
          <b><p><i>Input Image</i></p></b></br>
          <img src="images/image2.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <b><p><i>Output Image</i></p></b>
          <img src="images/image2new.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>
<pre>

person : 91.946941614151
--------------------------------
person : 73.61021637916565
--------------------------------
laptop : 90.24320840835571
--------------------------------
laptop : 73.6881673336029
--------------------------------
laptop : 95.16398310661316
--------------------------------
person : 87.10319399833679
--------------------------------

</pre>

<a href="imageai/Detection/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> 教程&文档 </button></a><br><br>

<div id="videodetection"></div>
<h3><b><u>视频对象检测和跟踪</u></b></h3>
<b>ImageAI</b> provides very convenient and powerful methods
在视频中执行对象检测并跟踪特定对象。 提供的视频对象检测类仅支持
  目前最先进的RetinaNet，但可以选择调整最先进的性能或实时处理。
您将在下面找到使用RetinaNet模型的示例对象检测结果，单击"教程和文档"链接以查看完整的示例代码。
<p><div style="width: 600px;" >
          <p><i><b>Video Object Detection</b></i></p>
<p><i>Below is a snapshot of a video with objects detected.</i></p>
          <img src="images/video1.jpg" style="width: 500px; height: auto; margin-left: 50px; " /> <br>
          <p><i><b>Video Custom Object Detection (Object Tracking)</b></i></p>
            <p><i>Below is a snapshot of a video with only person, bicycle and motorcyle detected.</i></p>
          <img src="images/video2.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>
</p>

<a href="imageai/Detection/VIDEO.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> 教程&文档 </button></a><br><br>

<div id="customtraining"></div>
<h3><b><u>自定义模型培训</u></b></h3>
<b>ImageAI</b> 为您提供类和方法，以训练可用于对您自己的自定义对象执行预测的新模型。
您可以使用SqueezeNet，ResNet50，InceptionV3和DenseNet以少于<b> 12 </b>行代码训练您的自定义模型。
单击"教程和文档"链接以查看完整的示例代码。
<br>
<p><br>
    <div style="width: 600px;" >
            <p><i>A sample from the IdenProf Dataset used to train a Model for predicting professionals.</i></p>
          <img src="images/idenprof.jpg" style="width: 500px; height: auto; margin-left: 50px; " />
    </div> <br>
</p>

<a href="imageai/Prediction/CUSTOMTRAINING.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a><br><br>

<div id="customprediction"></div>
<h3><b><u>自定义图像预测</u></b></h3>
<b>ImageAI</b> 为您提供类和方法，使用您自己训练的模型运行图像预测您自己的自定义对象 <b>ImageAI</b> Model Training class.
您可以使用使用SqueezeNet，ResNet50，InceptionV3和DenseNet训练的自定义模型以及包含自定义对象名称映射的JSON文件。
单击"教程和文档"链接以查看完整的示例代码。<br><p><br>
<p><i>从IdenProf培训的样本模型预测，用于预测专业人员</i></p>
      <img src="images/4.jpg" style="width: 400px; height: auto;" />
    <pre>mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575</pre>
</p>

<a href="imageai/Prediction/CUSTOMPREDICTION.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> 教程和文档 </button></a><br><br> <br>

<div id="performance"></div>
<h3><b><u>实时和高性能实施</u></b></h3>
<b>ImageAI</b> 提供最先进的计算机视觉技术的抽象和方便的实现. All of <b>ImageAI</b> 实现和代码可以在具有中等CPU容量的任何计算机系统上运行。 但是，CPU上的图像预测，对象检测等操作的处理速度很慢，不适合实时应用。 要以高性能执行实时计算机视觉操作，您需要使用支持GPU的技术。
<br> <br>
<b>ImageAI</b> 使用Tensorflow主干进行计算机视觉操作。 Tensorflow支持CPU和GPU（特别是NVIDIA GPU。您可以为您的PC获得一台或拥有一台PC），用于机器学习和人工智能算法的实施。 要使用支持使用GPU的Tensorflow，请点击以下链接 :
<br> <br>
FOR WINDOWS <br>
<a href="https://www.tensorflow.org/install/install_windows" >https://www.tensorflow.org/install/install_windows</a> <br><br>
FOR macOS <br>
<a href="https://www.tensorflow.org/install/install_mac" >https://www.tensorflow.org/install/install_mac</a> <br><br>
FOR UBUNTU <br>
<a href="https://www.tensorflow.org/install/install_linux">https://www.tensorflow.org/install/install_linux</a>
<br><br>

<div id="sample"></div>
<h3><b><u>示例应用程序</u></b></h3>
      作为使用ImageAI可以做什么的演示，我们为Windows构建了一个完整的AI驱动的照片库 <b>IntelliP</b> ,  using <b>ImageAI</b> and UI framework <b>Kivy</b>. Follow this 
 <a href="https://github.com/OlafenwaMoses/IntelliP"  > link </a> 下载应用程序的页面及其源代码. <br> <br>
我们也欢迎您提交的应用程序和系统的提交，并由ImageAI提供在此处列出的列表。 你想要你的 ImageAI powered 这里列出的发展，你可以通过我们的方式联系我们 <a href="#contact" >Contacts</a> below. <br> <br>

<div id="recommendation"></div>
 <h3><b><u>AI实践建议</u></b></h3>
对于任何有兴趣建立人工智能系统并将其用于商业，经济，社会和研究目的的人来说，至关重要的是该人员知道这些技术的使用可能产生的积极，消极和前所未有的影响。 他们还必须了解经验丰富的行业专家建议的方法和实践，以确保人工智能的每次使用都为人类带来整体利益。 因此，我们建议所有希望使用ImageAI和其他人工智能工具和资源的人阅读微软2018年1月出版的题为“未来计算：人工智能及其在社会中的作用”的出版物。
请点击以下链接下载该出版物。
 <br><br>
<a href="https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society/" >https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society/</a>
 <br><br>

<div id="contact"></div>
 <h3><b><u>联系开发人员</u></b></h3>
 <p> <b>Moses Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:guymodscientist@gmail.com"> guymodscientist@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://moses.specpal.science"> https://moses.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/OlafenwaMoses"> @OlafenwaMoses</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@guymodscientist"> @guymodscientist</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" target="_blank" href="https://facebook.com/moses.olafenwa"> moses.olafenwa</a> <br>
<br><br>
      <b>John Olafenwa</b> <br>
    <i>Email: </i>    <a style="text-decoration: none;"  href="mailto:johnolafenwa@gmail.com"> johnolafenwa@gmail.com</a> <br>
      <i>Website: </i>    <a style="text-decoration: none;" target="_blank" href="https://john.specpal.science"> https://john.specpal.science</a> <br>
      <i>Twitter: </i>    <a style="text-decoration: none;" target="_blank" href="https://twitter.com/johnolafenwa"> @johnolafenwa</a> <br>
      <i>Medium : </i>    <a style="text-decoration: none;" target="_blank" href="https://medium.com/@johnolafenwa"> @johnolafenwa</a> <br>
      <i>Facebook : </i>    <a style="text-decoration: none;" href="https://facebook.com/olafenwajohn"> olafenwajohn</a> <br>
 </p><br>

 <div id="ref"></div>
 <h3><b><u>参考</u></b></h3>

 1. Somshubra Majumdar，DenseNet论文的实施，Keras中密集连接的卷积网络 <br>
 <a href="https://github.com/titu1994/DenseNet/" >https://github.com/titu1994/DenseNet/</a> <br><br>

 2. 麻省理工学院和哈佛大学广泛研究所，Keras深度残留网络包 <br>
 <a href="https://github.com/broadinstitute/keras-resnet" >https://github.com/broadinstitute/keras-resnet</a> <br><br>

 3. Fizyr，Keras实施RetinaNet对象检测 <br>
 <a href="https://github.com/fizyr/keras-retinanet" >https://github.com/fizyr/keras-retinanet</a> <br><br>

 4. Francois Chollet，流行的deeplearning模型的Keras代码和权重文件 <br>
 <a href="https://github.com/fchollet/deep-learning-models" >https://github.com/fchollet/deep-learning-models</a> <br><br>

 5. Forrest N.等人，SqueezeNet：AlexNet级精度，参数减少50倍，模型尺寸小于0.5MB <br>
 <a href="https://arxiv.org/abs/1602.07360" >https://arxiv.org/abs/1602.07360</a> <br><br>

 6. Kaiming H. et al, Deep Residual Learning for Image Recognition <br>
 <a href="https://arxiv.org/abs/1512.03385" >https://arxiv.org/abs/1512.03385</a> <br><br>

 7. Szegedy 等，重新思考计算机视觉的初始架构 <br>
 <a href="https://arxiv.org/abs/1512.00567" >https://arxiv.org/abs/1512.00567</a> <br><br>

 8. Gao. et al, 密集连接的卷积网络 <br>
 <a href="https://arxiv.org/abs/1608.06993" >https://arxiv.org/abs/1608.06993</a> <br><br>

 9. Tsung-Yi. et al, 密集物体检测的焦点损失 <br>
 <a href="https://arxiv.org/abs/1708.02002" >https://arxiv.org/abs/1708.02002</a> <br><br>
 
 10. O Russakovsky et al, 大规模视觉识别挑战 <br>
 <a href="https://arxiv.org/abs/1409.0575" >https://arxiv.org/abs/1409.0575</a> <br><br>
 
 11. TY Lin et al, Microsoft COCO: 上下文中的常见对象 <br>
 <a href="https://arxiv.org/abs/1405.0312" >https://arxiv.org/abs/1405.0312</a> <br><br>
 
 12. Moses & John Olafenwa, 可识别专业人员的图像集合.<br>
 <a href="https://github.com/OlafenwaMoses/IdenProf" >https://github.com/OlafenwaMoses/IdenProf</a> <br><br>
 
 
 
