# TransRVNet



## Code coming Soon...Currently we only show our results for visualization and evaluation purposes.

## Requirements: vispy==0.5.3 PyYAML==5.1.1

## First please download the [SemanticPoss Dataset](http://www.poss.pku.edu.cn./download.html) from the official website.



## Then download our predictions from [Google Drive](https://drive.google.com/file/d/1i3GMG4KcOPwDbJtgZEDWJkSApOAnRn5O/view?usp=sharing).



## Visualize Example:


- Visualize GT:

  `python visualize.py -d /dataset -s sequences`

- Visualize Our Predictions:

  `python visualize.py -d /dataset -p /predictions -s 02/03`
- In the visualization:

      To navigate:

          b: back (previous scan)

          n: next (next scan)

          q: quit (exit program)

## Visualize Video:
[![Video](https://res.cloudinary.com/marcomontalbano/image/upload/v1627450328/video_to_markdown/images/youtube--NXyBSnMek3M-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/NXyBSnMek3M "Video")
## Eval Example:

- Eval Seq 02:

  `python evaluate_iou.py -d /dataset -p /predictions -s valid02` 

- Eval seq 03:

  `python evaluate_iou.py -d /dataset -p /predictions -s valid03`



## Acknowledgment

The code is partly based on [LiDAR-Bonnetal](https://github.com/PRBonn/lidar-bonnetal) and [SalsaNext](https://github.com/Halmstad-University/SalsaNext). Thanks for their open source work.

### Citation

Currently, please consider citing:

```
@inproceedings{pan2020semanticposs,
  title={Semanticposs: A point cloud dataset with large quantity of dynamic instances},
  author={Pan, Yancheng and Gao, Biao and Mei, Jilin and Geng, Sibo and Li, Chengkun and Zhao, Huijing},
  booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
  pages={687--693},
  year={2020},
  organization={IEEE}
}
```
