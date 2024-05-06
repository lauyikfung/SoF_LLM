### 高考作文数据集

+ standard.jsonl：手工标注20条评分对齐的数据
+ excellent.json：手工爬取人工撰写的满分作文或优秀作文29条
+ base.jsonl：爬取高考作文原题和模拟题约1.7k条
+ filtered_data_aug：扩充并筛选后的数据，约7.5k条，包含题目，参考作文和参考评分（均为kimi生成）
+ data_aug.json：扩充但未筛选的数据，约34k条，仅包含题目（均为kimi生成）