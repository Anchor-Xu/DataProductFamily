# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:35:47 2025

@author: 30743
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter.font import Font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CompleteProductConfigViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("产品配置优化完整结果展示")
        self.root.geometry("1400x900")
        
        # 完整参数数据（来自参数核对2.py）
        self.complete_params = {
            'markets': 5,
            'product_families': 4,
            'composite_modules': 8,
            'basic_modules': 12,
            'tech_info_modules': 21,
            'options': 42,
            'suppliers': 30,
            'mandatory_F': [2, 5, 7],  # F3, F6, F8
            'common_FM': [2, 3, 5, 7],  # FM3, FM4, FM6, FM8
            'best_fitness': 3.0714,
            'total_utility': 132317.41,
            'total_cost': 43080.39,
            'supplier_count': 24,
            'convergence_gen': 11,
            'total_time': 3868.30,
            'f_fm_map': {
                0: [2, 3, 6, 8],   # F1: FM3,FM4,FM7,FM9
                1: [2, 4, 7, 9],   # F2: FM3,FM5,FM8,FM10
                2: [1, 2, 3, 5],   # F3: FM2,FM3,FM4,FM6
                3: [2, 3, 6, 8],   # F4: FM3,FM4,FM7,FM9
                4: [1, 5, 10, 11], # F5: FM2,FM6,FM11,FM12
                5: [0, 2, 3, 4],   # F6: FM1,FM3,FM4,FM5
                6: [5, 7, 9, 10],  # F7: FM6,FM8,FM10,FM11
                7: [4, 5, 7, 10]   # F8: FM5,FM6,FM8,FM11
            },
            'fm_dmam_map': {
                0: [0, 1, 2, 9],    # FM1: DMAM1,2,3,10
                1: [1, 11],         # FM2: DMAM2,12
                2: [0, 6, 13, 14],  # FM3: DMAM1,7,14,15
                3: [3, 4, 15],      # FM4: DMAM4,5,16
                4: [0, 2, 5, 16],   # FM5: DMAM1,3,6,17
                5: [2, 4, 10],      # FM6: DMAM3,5,11
                6: [5, 12],         # FM7: DMAM6,13
                7: [4, 6, 10, 17],  # FM8: DMAM5,7,11,18
                8: [7, 10, 18],     # FM9: DMAM8,11,19
                9: [6, 10, 19],     # FM10: DMAM7,11,20
                10: [4, 10],        # FM11: DMAM5,11
                11: [4, 8, 9, 20]   # FM12: DMAM5,9,10,21
            },
            'supplier_names': {
                0: "Google", 1: "阿里云PAI", 2: "华为ModelArts", 3: "Microsoft",
                4: "腾讯云", 5: "蚂蚁OceanBase", 6: "Apache Flink", 7: "DataRobot",
                8: "NVIDIA", 9: "Hugging Face", 10: "国家企业信用信息公示系统",
                11: "中国人民银行征信中心", 12: "中国裁判文书网", 13: "信用中国",
                14: "人力资源和社会保障部", 15: "自然资源部", 16: "地方登记中心",
                17: "各商业银行", 18: "四大会计师事务所", 19: "国家级行业协会",
                20: "行业会员企业", 21: "商业信息平台", 22: "市场调研机构",
                23: "法律数据库", 24: "商业征信平台", 25: "官方媒体",
                26: "自媒体平台", 27: "企业ERP系统", 28: "电商平台", 29: "国家统计局"
            }
        }
        
        # 完整配置数据（来自结果1(1).docx）
        self.complete_configs = {
            "产品族1": {
                1: {  # F2（可选）
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (27, 12), # DMAM14: 选项28 (法律数据库)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    7: {  # FM8（常用）
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        6: (13, 9),  # DMAM7: 选项14 (Hugging Face)
                        10: (20, 27), # DMAM11: 选项21 (企业ERP系统)
                        17: (35, 18)  # DMAM18: 选项36 (四大会计师事务所)
                    },
                    9: {  # FM10（可选）
                        6: (13, 9),  # DMAM7: 选项14 (Hugging Face)
                        10: (21, 22), # DMAM11: 选项22 (四大会计师事务所)
                        19: (38, 19)  # DMAM20: 选项40 (地方登记中心)
                    }
                },
                2: {  # F3（必选）
                    1: {  # FM2（可选）
                        1: (3, 3),   # DMAM2: 选项4 (Microsoft)
                        11: (22, 29) # DMAM12: 选项23 (国家统计局)
                    },
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 9),  # DMAM7: 选项14 (Hugging Face)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        15: (30, 16) # DMAM16: 选项32 (商业征信平台)
                    },
                    5: {  # FM6（常用）
                        2: (5, 2),   # DMAM3: 选项6 (阿里云PAI)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    }
                },
                5: {  # F6（必选）
                    0: {  # FM1（可选）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        1: (2, 3),   # DMAM2: 选项3 (Microsoft)
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        9: (18, 10)  # DMAM10: 选项20 (国家企业信用信息公示系统)
                    },
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 9),  # DMAM7: 选项14 (Hugging Face)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    }
                },
                6: {  # F7（可选）
                    5: {  # FM6（常用）
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    },
                    7: {  # FM8（常用）
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        10: (21, 22), # DMAM11: 选项22 (四大会计师事务所)
                        17: (35, 18)  # DMAM18: 选项36 (四大会计师事务所)
                    }
                },
                7: {  # F8（必选）
                    5: {  # FM6（常用）
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (21, 22) # DMAM11: 选项22 (四大会计师事务所)
                    }
                }
            },
            "产品族2": {
                1: {  # F2（可选）
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (27, 12), # DMAM14: 选项28 (法律数据库)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    4: {  # FM5（可选）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        5: (11, 7),  # DMAM6: 选项12 (DataRobot)
                        16: (33, 26) # DMAM17: 选项34 (自媒体平台)
                    },
                    7: {  # FM8（常用）
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        10: (21, 22), # DMAM11: 选项22 (四大会计师事务所)
                        17: (35, 18)  # DMAM18: 选项36 (四大会计师事务所)
                    },
                    9: {  # FM10（可选）
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        10: (21, 22), # DMAM11: 选项22 (四大会计师事务所)
                        19: (38, 19)  # DMAM20: 选项40 (国家级行业协会)
                    }
                },
                2: {  # F3（必选）
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (27, 12), # DMAM14: 选项28 (法律数据库)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    3: {  # FM4（常用）
                        3: (6, 6),   # DMAM4: 选项7 (Apache Flink)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        15: (30, 16) # DMAM16: 选项31 (中国人民银行征信中心)
                    },
                    5: {  # FM6（常用）
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (21, 22) # DMAM11: 选项22 (四大会计师事务所)
                    }
                },
                3: {  # F4（可选）
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        15: (30, 16) # DMAM16: 选项31 (中国人民银行征信中心)
                    }
                },
                4: {  # F5（可选）
                    1: {  # FM2（可选）
                        1: (2, 3),   # DMAM2: 选项3 (Microsoft)
                        11: (22, 29) # DMAM12: 选项23 (国家统计局)
                    },
                    5: {  # FM6（常用）
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    }
                },
                5: {  # F6（必选）
                    0: {  # FM1（可选）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        1: (2, 3),   # DMAM2: 选项3 (Microsoft)
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        9: (18, 10)  # DMAM10: 选项20 (国家企业信用信息公示系统)
                    },
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    3: {  # FM4（常用）
                        3: (6, 6),   # DMAM4: 选项7 (Apache Flink)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        15: (30, 16) # DMAM16: 选项31 (中国人民银行征信中心)
                    },
                    4: {  # FM5（可选）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        5: (11, 7),  # DMAM6: 选项12 (DataRobot)
                        16: (32, 26) # DMAM17: 选项33 (官方媒体)
                    }
                },
                6: {  # F7（可选）
                    5: {  # FM6（常用）
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    },
                    9: {  # FM10（可选）
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        10: (20, 27), # DMAM11: 选项21 (企业ERP系统)
                        19: (39, 16)  # DMAM20: 选项40 (地方登记中心)
                    }
                },
                7: {  # F8（必选）
                    4: {  # FM5（可选）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        5: (10, 7),  # DMAM6: 选项11 (DataRobot)
                        16: (32, 26) # DMAM17: 选项33 (官方媒体)
                    },
                    10: {  # FM11（可选）
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        10: (21, 22) # DMAM11: 选项22 (四大会计师事务所)
                    }
                }
            },
            "产品族3": {
                0: {  # F1（可选）
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (12, 9),  # DMAM7: 选项13 (Hugging Face)
                        13: (27, 12), # DMAM14: 选项28 (法律数据库)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        15: (31, 16) # DMAM16: 选项32 (商业征信平台)
                    }
                },
                2: {  # F3（必选）
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    5: {  # FM6（常用）
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    }
                },
                3: {  # F4（可选）
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (12, 9),  # DMAM7: 选项13 (Hugging Face)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    3: {  # FM4（常用）
                        3: (6, 6),   # DMAM4: 选项7 (Apache Flink)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        15: (30, 16) # DMAM16: 选项31 (中国人民银行征信中心)
                    },
                    8: {  # FM9（可选）
                        7: (15, 5),  # DMAM8: 选项16 (蚂蚁OceanBase)
                        10: (21, 22), # DMAM11: 选项22 (四大会计师事务所)
                        18: (37, 19)  # DMAM19: 选项38 (国家级行业协会)
                    }
                },
                5: {  # F6（必选）
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (12, 9),  # DMAM7: 选项13 (Hugging Face)
                        13: (27, 12), # DMAM14: 选项28 (法律数据库)
                        14: (28, 13)  # DMAM15: 选项29 (信用中国)
                    },
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        15: (31, 16) # DMAM16: 选项32 (商业征信平台)
                    }
                },
                7: {  # F8（必选）
                    5: {  # FM6（常用）
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    }
                }
            },
            "产品族4": {
                0: {  # F1（可选）
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        15: (31, 16) # DMAM16: 选项32 (商业征信平台)
                    }
                },
                2: {  # F3（必选）
                    2: {  # FM3（常用）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        6: (12, 9),  # DMAM7: 选项13 (Hugging Face)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    3: {  # FM4（常用）
                        3: (6, 6),   # DMAM4: 选项7 (Apache Flink)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        15: (31, 16) # DMAM16: 选项32 (商业征信平台)
                    },
                    5: {  # FM6（常用）
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (21, 22) # DMAM11: 选项22 (四大会计师事务所)
                    }
                },
                5: {  # F6（必选）
                    2: {  # FM3（常用）
                        0: (0, 0),   # DMAM1: 选项1 (Google)
                        6: (13, 3),  # DMAM7: 选项14 (Microsoft)
                        13: (26, 12), # DMAM14: 选项27 (中国裁判文书网)
                        14: (29, 21)  # DMAM15: 选项30 (商业信息平台)
                    },
                    3: {  # FM4（常用）
                        3: (7, 6),   # DMAM4: 选项8 (腾讯云)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        15: (30, 16) # DMAM16: 选项31 (中国人民银行征信中心)
                    },
                    4: {  # FM5（可选）
                        0: (1, 1),   # DMAM1: 选项2 (阿里云PAI)
                        2: (4, 2),   # DMAM3: 选项5 (华为ModelArts)
                        5: (10, 7),  # DMAM6: 选项11 (DataRobot)
                        16: (32, 26) # DMAM17: 选项33 (官方媒体)
                    }
                },
                7: {  # F8（必选）
                    5: {  # FM6（常用）
                        2: (5, 1),   # DMAM3: 选项6 (阿里云PAI)
                        4: (9, 0),   # DMAM5: 选项10 (Google)
                        10: (20, 27) # DMAM11: 选项21 (企业ERP系统)
                    },
                    10: {  # FM11（可选）
                        4: (8, 0),   # DMAM5: 选项9 (Google)
                        10: (21, 22) # DMAM11: 选项22 (四大会计师事务所)
                    }
                }
            }
        }
        
        # 灵敏度分析数据
        self.sensitivity_data = {
            'mutation_rate': {
                'values': [0.1, 0.2, 0.3, 0.4, 0.5],
                'fitness': [3.5614, 3.3177, 3.2236, 2.9308, 3.4721],
                'utility': [120139.00, 101361.22, 110245.66, 123154.86, 131740.16],
                'cost': [45093.17, 44522.99, 50963.19, 56412.96, 55301.16],
                'time': [1083.95, 1083.91, 1102.51, 1103.50, 1107.88],
                'suppliers': [30, 30, 30, 30, 30]
            },
            'market_demand': {
                'values': [0.5, 0.75, 1.0, 1.25, 1.5],
                'fitness': [1.4970, 2.4623, 3.2258, 3.8757, 4.9739],
                'utility': [64920.49, 99960.47, 130637.64, 145572.42, 200675.71],
                'cost': [53699.28, 55912.54, 55404.58, 51744.42, 57671.04],
                'time': [1282.13, 1828.70, 1281.03, 1098.08, 1638.64],
                'suppliers': [30, 30, 30, 30, 30]
            },
            'crossover_prob': {
                'values': [0.6, 0.7, 0.8, 0.9, 1.0],
                'fitness': [3.0460, 3.0982, 3.2641, 2.9647, 3.3692],
                'utility': [111180.50, 118981.20, 134355.54, 121155.36, 116201.65],
                'cost': [48681.17, 53211.67, 58138.64, 55935.82, 54351.44],
                'time': [1276.67, 1284.81, 1661.04, 1096.29, 1094.69],
                'suppliers': [30, 30, 30, 30, 30]
            },
            'mnl_param': {
                'values': [0.5, 0.75, 1.0, 1.25, 1.5],
                'fitness': [2.6359, 3.2283, 3.2846, 3.3739, 3.4207],
                'utility': [123848.10, 118561.09, 124695.35, 130195.21, 128934.83],
                'cost': [56723.94, 54171.39, 56479.13, 57876.78, 57678.73],
                'time': [1120.83, 1462.65, 3231.87, 1093.12, 1825.65],
                'suppliers': [30, 30, 30, 30, 30]
            }
        }
        
        # 灵敏度分析结论
        self.sensitivity_conclusions = {
            'mutation_rate': (
                "变异率灵敏度分析结论：\n"
                "• 变异率0.1时适应度最佳(3.56)，0.4时最低(2.93)\n"
                "• 总效用随变异率上升而增长，0.5时达到峰值131740\n"
                "• 运行时间相对稳定(1083-1108秒)，算法稳定性良好\n"
                "• 建议选择变异率0.1以获得最佳解质量与成本效益"
            ),
            'market_demand': (
                "市场需求灵敏度分析结论：\n"
                "• 市场需求与性能呈显著正相关关系\n"
                "• 适应度从0.5时的1.50增长至1.5时的4.97(增幅231%)\n"
                "• 总效用从64920提升至200675(增长209%)\n"
                "• 成本增幅相对平缓(仅7.4%)，显示良好规模效应"
            ),
            'crossover_prob': (
                "交叉概率灵敏度分析结论：\n"
                "• 交叉概率0.8时适应度最佳(3.26)，较0.9时高出10.1%\n"
                "• 总效用同样在0.8时取得最大值134355\n"
                "• 运行时间在0.8时出现峰值1661秒，收敛难度增加\n"
                "• 建议采用0.7-0.8的折中方案平衡性能与效率"
            ),
            'mnl_param': (
                "MNL模型参数灵敏度分析结论：\n"
                "• 适应度随参数增大线性增长(2.64→3.42)\n"
                "• 总效用在1.25时达到峰值130195\n"
                "• 总成本保持相对稳定(波动范围仅56724-57877)\n"
                "• 参数1.0时运行时间异常峰值3231秒\n"
                "• 建议选择1.25参数设置规避计算时间风险"
            )
        }
        
        # 创建界面
        self._setup_complete_ui()
    
    def _setup_complete_ui(self):
        # 主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部标题栏
        self._create_complete_header()
        
        # 主内容区
        self._create_complete_content()
        
        # 底部状态栏
        self._create_complete_footer()
        
        # 加载完整数据
        self._load_complete_data()
    
    def _create_complete_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame,
            text="产品配置优化完整结果展示与灵敏度分析系统",
            font=('Helvetica', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # 完整参数摘要
        params_text = (
            f"市场: {self.complete_params['markets']} | "
            f"产品族: {self.complete_params['product_families']} | "
            f"复合模块: {self.complete_params['composite_modules']} | "
            f"基本模块: {self.complete_params['basic_modules']} | "
            f"数据模块: {self.complete_params['tech_info_modules']} | "
            f"选项: {self.complete_params['options']} | "
            f"供应商: {self.complete_params['suppliers']}"
        )
        params_label = ttk.Label(header_frame, text=params_text)
        params_label.pack(side=tk.RIGHT)
    
    def _create_complete_content(self):
        # 主内容容器
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧导航树 (30%)
        tree_frame = ttk.Frame(content_frame, width=300)
        tree_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.tree = ttk.Treeview(tree_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # 右侧详情区 (70%)
        detail_frame = ttk.Frame(content_frame)
        detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # 详情选项卡
        self.notebook = ttk.Notebook(detail_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 配置详情选项卡
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="完整配置详情")
        
        self.detail_text = scrolledtext.ScrolledText(
            self.config_tab,
            wrap=tk.WORD,
            font=('Consolas', 10),
            padx=10,
            pady=10
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        
        # 优化结果选项卡
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="完整优化结果")
        
        self._create_complete_result_tab()
        
        # 灵敏度分析选项卡
        self.sensitivity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sensitivity_tab, text="灵敏度分析")
        
        self._create_sensitivity_tab()
    
    def _create_complete_result_tab(self):
        """创建完整优化结果选项卡"""
        # 使用网格布局展示关键指标
        metrics_frame = ttk.Frame(self.result_tab)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        # 完整指标卡片
        metrics = [
            ("最佳适应度", f"{self.complete_params['best_fitness']:.4f}", "#4CAF50"),
            ("总效用", f"{self.complete_params['total_utility']:,.2f}", "#2196F3"),
            ("总成本", f"{self.complete_params['total_cost']:,.2f}", "#F44336"),
            ("供应商数量", self.complete_params['supplier_count'], "#9C27B0"),
            ("收敛代数", self.complete_params['convergence_gen'], "#FF9800"),
            ("计算时间", f"{self.complete_params['total_time']:,.2f}秒", "#607D8B")
        ]
        
        for i, (title, value, color) in enumerate(metrics):
            card = ttk.Frame(metrics_frame, relief=tk.RAISED, borderwidth=1)
            card.grid(row=i//3, column=i%3, padx=5, pady=5, sticky="nsew")
            
            ttk.Label(card, text=title, font=('Helvetica', 10)).pack(pady=(5, 0))
            ttk.Label(
                card,
                text=value,
                font=('Helvetica', 14, 'bold'),
                foreground=color
            ).pack(pady=(0, 5))
        
        # 添加完整配置摘要
        summary_frame = ttk.Frame(self.result_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        summary_label = ttk.Label(
            summary_frame,
            text="完整配置摘要",
            font=('Helvetica', 12, 'bold')
        )
        summary_label.pack(anchor=tk.W)
        
        summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            height=10
        )
        summary_text.pack(fill=tk.BOTH, expand=True)
        
        # 填充完整摘要数据
        summary_text.insert(tk.END, "必选复合模块使用情况:\n")
        for f in self.complete_params['mandatory_F']:
            summary_text.insert(tk.END, f"• F{f+1}: 所有产品族均已配置\n")
        
        summary_text.insert(tk.END, "\n常用基本模块使用情况:\n")
        for fm in self.complete_params['common_FM']:
            summary_text.insert(tk.END, f"• FM{fm+1}: 高频率使用\n")
        
        summary_text.insert(tk.END, "\n供应商使用情况:\n")
        summary_text.insert(tk.END, f"• 共使用 {self.complete_params['supplier_count']} 家供应商\n")
        
        # 统计供应商使用频率
        supplier_usage = {}
        for family_config in self.complete_configs.values():
            for f_config in family_config.values():
                for fm_config in f_config.values():
                    for dmam_config in fm_config.values():
                        supplier_id = dmam_config[1]
                        supplier_usage[supplier_id] = supplier_usage.get(supplier_id, 0) + 1
        
        # 显示最常用的5家供应商
        top_suppliers = sorted(supplier_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        summary_text.insert(tk.END, "• 最常用供应商: ")
        summary_text.insert(tk.END, ", ".join(
            [f"{self.complete_params['supplier_names'][s[0]]}({s[1]}次)" 
             for s in top_suppliers]
        ))
        summary_text.insert(tk.END, "\n")
        
        summary_text.config(state=tk.DISABLED)
        
    def _create_sensitivity_tab(self):
        """创建灵敏度分析选项卡"""
        # 创建参数选择框架
        param_frame = ttk.Frame(self.sensitivity_tab)
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text="选择分析参数:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        self.param_var = tk.StringVar(value="mutation_rate")
        param_combo = ttk.Combobox(
            param_frame,
            textvariable=self.param_var,
            values=["mutation_rate", "market_demand", "crossover_prob", "mnl_param"],
            state="readonly",
            width=15
        )
        param_combo.pack(side=tk.LEFT, padx=5)
        param_combo.bind("<<ComboboxSelected>>", self._update_sensitivity_display)
        
        # 创建图表和数据分析框架
        analysis_frame = ttk.Frame(self.sensitivity_tab)
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 左侧图表区域
        self.chart_frame = ttk.Frame(analysis_frame, width=600)
        self.chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 右侧数据分析区域
        data_frame = ttk.Frame(analysis_frame, width=300)
        data_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # 数据表格
        ttk.Label(data_frame, text="详细数据", font=('Helvetica', 11, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.data_text = scrolledtext.ScrolledText(
            data_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            height=15
        )
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        # 分析结论
        ttk.Label(data_frame, text="分析结论", font=('Helvetica', 11, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        self.conclusion_text = scrolledtext.ScrolledText(
            data_frame,
            wrap=tk.WORD,
            font=('Helvetica', 10),
            height=8
        )
        self.conclusion_text.pack(fill=tk.BOTH, expand=True)
        
        # 初始显示
        self._update_sensitivity_display()
        
    def _update_sensitivity_display(self, event=None):
        """更新灵敏度分析显示"""
        param_type = self.param_var.get()
        data = self.sensitivity_data[param_type]
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 清除之前的图表
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # 创建新图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.tight_layout(pad=3.0)
        
        # 适应度图表
        ax1.plot(data['values'], data['fitness'], 'o-', color='#4CAF50', linewidth=2)
        ax1.set_xlabel(self._get_param_label(param_type))
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度变化')
        ax1.grid(True, alpha=0.3)
        
        # 效用图表
        ax2.plot(data['values'], data['utility'], 'o-', color='#2196F3', linewidth=2)
        ax2.set_xlabel(self._get_param_label(param_type))
        ax2.set_ylabel('总效用')
        ax2.set_title('总效用变化')
        ax2.grid(True, alpha=0.3)
        
        # 成本图表
        ax3.plot(data['values'], data['cost'], 'o-', color='#F44336', linewidth=2)
        ax3.set_xlabel(self._get_param_label(param_type))
        ax3.set_ylabel('总成本')
        ax3.set_title('总成本变化')
        ax3.grid(True, alpha=0.3)
        
        # 时间图表
        ax4.plot(data['values'], data['time'], 'o-', color='#FF9800', linewidth=2)
        ax4.set_xlabel(self._get_param_label(param_type))
        ax4.set_ylabel('运行时间(秒)')
        ax4.set_title('运行时间变化')
        ax4.grid(True, alpha=0.3)
        
        # 嵌入图表到界面
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 更新数据表格
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        
        param_label = self._get_param_label(param_type)
        self.data_text.insert(tk.END, f"{param_label}\t适应度\t总效用\t总成本\t运行时间\t供应商\n")
        self.data_text.insert(tk.END, "-" * 80 + "\n")
        
        for i in range(len(data['values'])):
            self.data_text.insert(tk.END, 
                f"{data['values'][i]}\t{data['fitness'][i]:.4f}\t"
                f"{data['utility'][i]:.2f}\t{data['cost'][i]:.2f}\t"
                f"{data['time'][i]:.2f}\t{data['suppliers'][i]}\n"
            )
        
        self.data_text.config(state=tk.DISABLED)
        
        # 更新分析结论
        self.conclusion_text.config(state=tk.NORMAL)
        self.conclusion_text.delete(1.0, tk.END)
        self.conclusion_text.insert(tk.END, self.sensitivity_conclusions[param_type])
        self.conclusion_text.config(state=tk.DISABLED)
    
    def _get_param_label(self, param_type):
        """获取参数类型的中文标签"""
        labels = {
            'mutation_rate': '变异率',
            'market_demand': '市场需求',
            'crossover_prob': '交叉概率',
            'mnl_param': 'MNL模型参数'
        }
        return labels.get(param_type, param_type)
    
    def _create_complete_footer(self):
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_label = ttk.Label(
            footer_frame,
            text=f"优化完成 | 收敛代数: {self.complete_params['convergence_gen']} | "
                 f"总耗时: {self.complete_params['total_time']:.2f}秒"
        )
        status_label.pack(side=tk.LEFT)
        
        copyright_label = ttk.Label(
            footer_frame,
            text="© 2025 产品优化系统 - 完整结果与灵敏度分析"
        )
        copyright_label.pack(side=tk.RIGHT)
    
    def _load_complete_data(self):
        """加载完整数据到树形视图"""
        # 产品族数据
        for family_name in ["产品族1", "产品族2", "产品族3", "产品族4"]:
            family_node = self.tree.insert("", tk.END, text=family_name, open=False)
            
            # 添加该产品族的复合模块
            family_config = self.complete_configs.get(family_name, {})
            for f_idx in sorted(family_config.keys()):
                f_name = f"F{f_idx+1}"
                if f_idx in self.complete_params['mandatory_F']:
                    f_name += "（必选）"
                else:
                    f_name += "（可选）"
                
                f_node = self.tree.insert(family_node, tk.END, text=f_name, open=False)
                
                # 添加该复合模块的基本模块
                f_config = family_config[f_idx]
                for fm_idx in sorted(f_config.keys()):
                    fm_name = f"FM{fm_idx+1}"
                    if fm_idx in self.complete_params['common_FM']:
                        fm_name += "（常用）"
                    
                    self.tree.insert(f_node, tk.END, text=fm_name)
        
        # 绑定选择事件
        self.tree.bind("<<TreeviewSelect>>", self._on_complete_select)
        
        # 初始显示优化结果
        self._show_complete_results()
    
    def _on_complete_select(self, event):
        """处理树形视图选择事件"""
        selected_item = self.tree.focus()
        item_text = self.tree.item(selected_item, "text")

        if "产品族" in item_text:
            self._show_complete_family_detail(item_text)
        elif item_text.startswith("F") and "（可" in item_text:  # 复合模块节点
            self._show_complete_module_detail(item_text)
        elif item_text.startswith("FM"):  # 基本模块节点
            self._show_complete_fm_detail(item_text)

    
    def _show_complete_results(self):
        """显示完整优化结果"""
        self.notebook.select(self.result_tab)
    
    def _show_complete_family_detail(self, family_name):
        """显示完整产品族详情"""
        self.notebook.select(self.config_tab)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        
        self.detail_text.insert(tk.END, f"{family_name} 完整配置详情\n\n", "title")
        
        # 获取该产品族的完整配置
        family_config = self.complete_configs.get(family_name, {})
        
        for f_idx in sorted(family_config.keys()):
            f_name = f"F{f_idx+1}"
            if f_idx in self.complete_params['mandatory_F']:
                f_name += "（必选复合模块）"
            else:
                f_name += "（可选复合模块）"
            
            self.detail_text.insert(tk.END, f"{f_name}:\n")
            
            f_config = family_config[f_idx]
            for fm_idx in sorted(f_config.keys()):
                fm_name = f"FM{fm_idx+1}"
                if fm_idx in self.complete_params['common_FM']:
                    fm_name += "（常用基本模块）"
                
                self.detail_text.insert(tk.END, f"• {fm_name}:\n")
                
                fm_config = f_config[fm_idx]
                for dmam_idx in sorted(fm_config.keys()):
                    op_idx, supplier_id = fm_config[dmam_idx]
                    op_name = f"选项{op_idx+1}"
                    supplier_name = self.complete_params['supplier_names'].get(supplier_id, f"供应商{supplier_id}")
                    self.detail_text.insert(tk.END, f"  - DMAM{dmam_idx+1}: {op_name}（供应商：{supplier_name}）\n")
                
                self.detail_text.insert(tk.END, "\n")
        
        self.detail_text.config(state=tk.DISABLED)
    
    def _show_complete_module_detail(self, module_name):
        """显示完整复合模块详情"""
        self.notebook.select(self.config_tab)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        self.detail_text.insert(tk.END, f"{module_name} 完整详情\n\n", "title")

        # 提取模块编号 - 更健壮的方式
        try:
            # 从类似 "F2（必选）" 或 "F3（可选）" 的文本中提取数字
            f_idx = int(''.join(filter(str.isdigit, module_name.split('（')[0][1:]))) - 1
        except (ValueError, IndexError):
            self.detail_text.insert(tk.END, "无法解析模块编号\n")
            self.detail_text.config(state=tk.DISABLED)
            return

        # 显示模块基本信息
        if f_idx in self.complete_params['mandatory_F']:
            self.detail_text.insert(tk.END, "类型: 必选复合模块\n")
        else:
            self.detail_text.insert(tk.END, "类型: 可选复合模块\n")

        # 显示包含的基本模块
        self.detail_text.insert(tk.END, "\n包含基本模块:\n")
        for fm_idx in self.complete_params['f_fm_map'].get(f_idx, []):
            fm_name = f"FM{fm_idx+1}"
            if fm_idx in self.complete_params['common_FM']:
                fm_name += "（常用）"
            self.detail_text.insert(tk.END, f"• {fm_name}\n")

        # 统计使用该模块的产品族数量
        usage_count = 0
        for family_config in self.complete_configs.values():
            if f_idx in family_config:
                usage_count += 1

        self.detail_text.insert(tk.END, f"\n使用情况: {usage_count}个产品族配置\n")

        self.detail_text.config(state=tk.DISABLED)
    
    def _show_complete_fm_detail(self, fm_name):
        """显示完整基本模块详情"""
        self.notebook.select(self.config_tab)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        self.detail_text.insert(tk.END, f"{fm_name} 完整详情\n\n", "title")

        # 提取模块编号 - 更健壮的方式
        try:
            # 从类似 "FM3（常用）" 或 "FM4" 的文本中提取数字
            fm_idx = int(''.join(filter(str.isdigit, fm_name.split('（')[0][2:]))) - 1
        except (ValueError, IndexError):
            self.detail_text.insert(tk.END, "无法解析模块编号\n")
            self.detail_text.config(state=tk.DISABLED)
            return

        # 显示模块基本信息
        if fm_idx in self.complete_params['common_FM']:
            self.detail_text.insert(tk.END, "类型: 常用基本模块\n")
        else:
            self.detail_text.insert(tk.END, "类型: 可选基本模块\n")

        # 统计使用该模块的产品族数量
        usage_count = 0
        for family_config in self.complete_configs.values():
            for f_config in family_config.values():
                if fm_idx in f_config:
                    usage_count += 1
                    break

        self.detail_text.insert(tk.END, f"使用频率: {usage_count}个产品族\n\n")

        # 显示关联的技术信息模块
        self.detail_text.insert(tk.END, "关联数据模块:\n")
        for dmam_idx in self.complete_params['fm_dmam_map'].get(fm_idx, []):
            self.detail_text.insert(tk.END, f"• DMAM{dmam_idx+1}（数据模块）\n")

        self.detail_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = CompleteProductConfigViewer(root)
    root.mainloop()