# UniPose

  <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Artacho_UniPose_Unified_Human_Pose_Estimation_in_Single_Images_and_Videos_CVPR_2020_paper.html">**UniPose: Unified Human Pose Estimation in Single Images and Videos**</a>.
</p><br />

**NEW!: BAPose: Bottom-Up Pose Estimation with Disentangled Waterfall Representations**
<p align="justify">
Our novel framework for bottom-up multi-person pose estimation achieves State-of-the-Art results in several datasets. The pre-print of our new method, BAPose, can be found in the following link: <a href="https://arxiv.org/abs/2112.10716">BAPose pre-print</a>. Full code for the BAPose framework is scheduled to be released in the near future.
</p><br />

**NEW!: UniPose+: A unified framework for 2D and 3D human pose estimation in images and videos**
<p align="justify">
Our novel and improved UniPose+ framework for pose estimation achieves State-of-the-Art results in several datasets. UniPose+ can be found in the following link: <a href="https://www.computer.org/csdl/journal/tp/5555/01/09599531/1yeC9DHjMw8">UniPose+ at PAMI</a>. Full code for the UniPose+ framework is scheduled to be released in the near future.
</p><br />

**NEW!: OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation**
<p align="justify">
Our novel framework for multi-person pose estimation achieves State-of-the-Art results in several datasets. The pre-print of our new method, OmniPose, can be found in the following link: <a href="https://arxiv.org/abs/2103.10180">OmniPose pre-print</a>. Full code for the OmniPose framework is scheduled to be released in the near future.
Github: <a href="https://github.com/bmartacho/OmniPose">https://github.com/bmartacho/OmniPose</a>.
</p><br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/Unipose_pipeline.png" title="WASPnet architecture for Semantic Segmentation">
  Figure 1: UniPose architecture for single frame pose detection. The input color image of dimensions (HxW) is fed through the ResNet backbone and WASP module to obtain 256 feature channels at reduced resolution by a factor of 8. The decoder module generates K heatmaps, one per joint, at the original resolution, and the locations of the joints are determined by a local max operation.
</p><br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/Unipose_LSTM.png" title="WASPnet architecture for Semantic Segmentation">
  Figure 2: UniPose-LSTM architecture for pose estimation in videos. The joint heatmaps from the decoder of UniPose are fed into the LSTM along with the final heatmaps from the previous LSTM state. The convolutional layers following the LSTM reorganize the outputs into the final heatmaps used for joint localization.
</p><br />


<p align="justify">
We propose UniPose, a unified framework for human pose estimation, based on our "Waterfall" Atrous Spatial Pooling architecture, that achieves state-of-art-results on several pose estimation metrics. UniPose incorporates contextual segmentation and joint localization to estimate the human pose in a single stage, with high accuracy, without relying on statistical postprocessing methods. The Waterfall module in UniPose leverages the efficiency of progressive filtering in the cascade architecture, while maintaining multi-scale fields-of-view comparable to spatial pyramid configurations. Additionally, our method is extended to UniPose-LSTM for multi-frame processing and achieves state-of-the-art results for temporal pose estimation in Video. Our results on multiple datasets demonstrate that UniPose, with a ResNet backbone and Waterfall module, is a robust and efficient architecture for pose estimation obtaining state-of-the-art results in single person pose detection for both single images and videos.
  
We propose the “Waterfall Atrous Spatial Pyramid” module, shown in Figure 3. WASP is a novel architecture with Atrous Convolutions that is able to leverage both the larger Field-of-View of the Atrous Spatial Pyramid Pooling configuration and the reduced size of the cascade approach.<br />

<p align="center">
  <img src="https://www.mdpi.com/sensors/sensors-19-05361/article_deploy/html/images/sensors-19-05361-g006.png" width=500 title="WASP module"><br />
  Figure 3: WASP Module.
</p><br />

Examples of the UniPose architecture for Pose Estimation are shown in Figures 4 for single images and videos.<br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/supplemental.png" width=500 title="WASP module"><br />
  Figure 4: Pose estimation samples for UniPose in images and videos.
  <br /><br />
  
Link to the published article at <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Artacho_UniPose_Unified_Human_Pose_Estimation_in_Single_Images_and_Videos_CVPR_2020_paper.html">CVPR 2020</a>.
</p><br />

**Datasets:**
<p align="justify">
Datasets used in this paper and required for training, validation, and testing can be downloaded directly from the dataset websites below:<br />
  LSP Dataset: https://sam.johnson.io/research/lsp.html<br />
  MPII Dataset: http://human-pose.mpi-inf.mpg.de/<br />
  PennAction Dataset: http://dreamdragon.github.io/PennAction/<br />
  BBC Pose Dataset: https://www.robots.ox.ac.uk/~vgg/data/pose/<br />
</p><br />

**Pre-trained Models:**
<p align="justify">
The pre-trained weights can be downloaded
  <a href="https://drive.google.com/drive/folders/1dPc7AayY2Pi3gjUURgozkuvlab5Vr-9n?usp=sharing">here</a>.
</p><br />


**Contact:**

<p align="justify">
Bruno Artacho:<br />
  E-mail: bmartacho@mail.rit.edu<br />
  Website: https://www.brunoartacho.com<br />
  
Andreas Savakis:<br />
  E-mail: andreas.savakis@rit.edu<br />
  Website: https://www.rit.edu/directory/axseec-andreas-savakis<br /><br />
</p>

**Citation:**

<p align="justify">
Artacho, B.; Savakis, A. UniPose: Unified Human Pose Estimation in Single Images and Videos. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. <br />

Latex:<br />
@InProceedings{Artacho_2020_CVPR,<br />
  title = {UniPose: Unified Human Pose Estimation in Single Images and Videos},<br />
  author = {Artacho, Bruno and Savakis, Andreas},<br />
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},<br />
  month = {June},<br />
  year = {2020},<br />
}<br />
</p>
