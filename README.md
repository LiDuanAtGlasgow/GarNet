# Garment Similarity Network (GarNet): A Continuous Perception Robotic Approach for Predicting Shapes and Visually Perceived Weights of Unseen Garments
## Introduction
We present in this paper a Garment Similarity Network (GarNet) that learns geometric and physical similarities between known garments by continuously observing a garment while a robot picks it up from a table. The aim is to capture and encode geometric and physical characteristics of a garment into a manifold where a decision can be carried out, such as predicting the garment's shape class and its visually perceived weight. Our approach features an early stop strategy, which means that GarNet does not need to observe the entire video sequence to make a prediction and maintain high prediction accuracy values. In our experiments, we find that GarNet achieves prediction accuracies of 98 \% for shape classification and 95% for predicting weights. We compare our approach with state-of-art methods, and we observe that our approach advances the state-of-art methods from 70.8% to 98% for shape classification.

## Video Demonstration
<iframe width="560" height="315" src="https://www.youtube.com/embed/Zh_3xcOrbmg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

--------------------------------------------------------------------------------------------
## Examples of Early Stop enabled by the GarNet in a Continuous Perception Way
The early-stop strategy allow the robotic system to make a prediction without observing the full video sequence, and therefore, enabling continous perception. As it can be observed in both plots, GarNet becomes confident over time and the early-stop strategy activates if 80% of decision points in the garment similarity map are within a correct category.

<img src="images/Paper-Continuous_Perception_Part1.png" width="500" height="500">
<img src="images/Paper-Continuous_Perception_Part2.png" width="500" height="500">

-----------------------------------------------------------------------------------------------
## Downloads
<img src="images/Page_Design_Paper.png" width="50" height="50"><img src="images/Page_Design_Code.png" width="50" height="50"><img src="images/Page_Design_Database.png" width="50" height="50">\
 [<a taget="_blank" title="Paper" href="https://www.overleaf.com/read/wbhmkkpbgmwb">Paper</a>][<a taget="_blank" title="Code" href="https://github.com/LiDuanAtGlasgow/GarNet">Code</a>][<a taget="_blank" title="Database" href="https://gla-my.sharepoint.com/:u:/g/personal/2168518d_student_gla_ac_uk/EQ8QtIrqcUlNtT0GvLG8kYMBqrPiGziLJLR1pGD4r1T01w?e=02b7mr">Database</a>]

## The Author
<img src='images/Li_Duan_Ken.jpg' width='200' height='150'>\
My name is Li Duan (Ken) and I am a 3rd-year PhD student @ University of Glasgow, Scotland. My interests include robot continuous perception and deformable object manipulation. Recently, I am working on investigating geometric and physical properties of fabrics and garments, from which I am keen on engineering methods to improve robot deformable object manipulations.\
I am willing to hear from our community, so your suggestions are welcomed. Please reach me at:\
Twitter: [@liduanglasgow](https://twitter.com/liduanglasgow)\
Email: <em>l.duan.1 at research.gla.ac.uk</em>
