# 曲线图与源数据

所有图提供450dpi PNG和可编辑文字的SVG。英文标签方便后续正式报告直接使用；CSV可用于重新绘图。

| 文件编号 | 内容 | 对应源数据 |
|---|---|---|
| 01 | 风险模型逐轮非对称平方loss | model1_tree_prefix_metrics.csv |
| 02 | 风险模型逐轮加权MAE与加权RMSE | model1_tree_prefix_metrics.csv |
| 03 | 2026逐月单步回测误差及基准 | model1_rolling_monthly_metrics.csv |
| 04 | 98个可行方案、7点前沿及M0775 | model2_designs.csv、model2_pareto_frontier.csv |
| 05 | M0775在六种假设情景下的利润与利润率 | model2_selected_scenarios.csv |

## 训练曲线的来源

原训练未保存逐轮evaluation日志。01、02利用冻结GBDT模型的前0至200棵树逐步预测，恢复每轮损失和指标，**没有重新训练或改变参数**。第0轮是原训练初始化均值，第200轮加回相同初始分数后的输出与原模型一致。

- Training：目标月份早于2026年8月的1214条真实历史样本。
- August test：8月12家航司，是测试诊断，**不是独立验证曲线**。
- 第200轮测试加权MAE 0.3476354269，加权RMSE 0.4901317708，单位件/千件。
- loss为原始预测的平均单样本非对称平方项，不含L2正则项；MAE/RMSE使用部署时的非负截断输出。
- `weighted_`指标按实际行李量加权；训练loss仍使用原训练等权设置。
- 测试曲线后段有轻微回升。这是诊断现象，没有据此重选轮数或修改冻结的200轮设计。
- 训练数据跨度与波动不同于8月12条测试记录，因此不能仅凭训练loss高于测试loss判断数据泄漏。
- 全部为单个已冻结模型/既有回测的点估计，无置信区间或多随机种子统计检验。

03来自多个按月重训模型，不是一份模型的epoch曲线。04、05是模型二的网格搜索和假设收益，不能称作训练收敛或实际国泰利润。前沿连线仅连接有限候选点，不表示中间任意值都已评估。

复现：安装 `requirements-viz.txt` 后运行 `python scripts/export_visualization_curves.py`。来源与模型哈希见provenance.json及checks.json。
