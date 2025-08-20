# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 19:03:17 2025

@author: asus
"""

# -*- coding: utf-8 -*-
"""
嵌套遗传算法优化器（无产品族-复合模块映射版）
层级关系：产品族 → F（复合模块） → FM（基本模块） → DMAM（技术信息模块） → 选项 → 供应商
特性：去除产品族对复合模块的映射限制，所有F可被任意产品族选用
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import time
import json
import os
class DataProductOptimizer:
    """基础数据产品优化器（无产品族-F映射版）"""
    def __init__(self, params: Dict):
        self.params = params
        # 核心参数（移除产品族-F映射相关限制）
        self.MARKETS = params.get('markets', 5)  # 市场数量
        self.FAMILY = params.get('product_families', 4)  # 产品族数量（仅作为分类维度）
        self.F = params.get('composite_modules', 8)  # F（复合模块）数量
        self.FM = params.get('basic_modules', 12)  # FM（基本模块）数量
        self.DMAM = params.get('tech_info_modules', 21)  # DMAM数量
        self.OP = params.get('options', 42)  # 选项数量
        self.S = params.get('suppliers', 30)  # 供应商数量
        
        # 关键定义：必选复合模块F和常用基本模块FM
        self.mandatory_F = [2, 5, 7]  # F3（索引2）、F6（索引5）、F8（索引7）为必选
        self.common_FM = [2, 3, 5, 7]  # FM3（索引2）、FM4（索引3）、FM6（索引5）、FM8（索引7）为常用
        
        # 外部参数映射（移除family_f_map）
        self.f_fm_map = params.get('f_fm_map', {})  # F→FM映射
        self.fm_dmam_map = params.get('fm_dmam_map', {})  # FM→DMAM映射
        self.dmam_op_map = params.get('dmam_op_map', {})  # DMAM→选项映射
        self.supplier_op_map = params.get('supplier_op_map', {})  # 供应商→选项映射
        self.manufacturing_cost = params.get('manufacturing_cost', {})  # (S编号, Opid编号)
        self.risk_cost = params.get('risk_cost', {})  # S编号
        
        # 初始化模型参数
        self.market_demand = np.random.uniform(100, 1000, size=self.MARKETS)  # 各市场需求
        self.utility = np.random.uniform(0.1, 1.0, size=(self.MARKETS, self.DMAM, self.OP))  # 市场特异性效用
        self.module_weight = np.random.uniform(0.5, 1.5, size=(self.F, self.FM))  # F-FM权重
        self.mnl_param = params.get('mu', 1.0)  # MNL模型参数（用于市场选择概率）
    
    def initialize_individual(self) -> Dict:
        """初始化个体（所有F对产品族开放）"""
        individual = {
            # x[产品族][F][FM][DMAM][OP]：产品族选用F的FM的DMAM的选项
            'x': np.zeros((self.FAMILY, self.F, self.FM, self.DMAM, self.OP)),
            # y[F][FM][DMAM][OP][S]：选项由供应商S提供
            'y': np.zeros((self.F, self.FM, self.DMAM, self.OP, self.S)),
            'supplier_count': np.zeros(self.F),  # 每个F的供应商数量
            'fitness': None,
            'total_utility': None,
            'total_cost': None
        }
        
        # 按层级初始化：产品族→F→FM→DMAM→选项→供应商
        # 关键修改：去除产品族对F的限制，所有F均可被任意产品族选用
        for fam in range(self.FAMILY):  # 产品族
            for f in range(self.F):  # 所有复合模块F均可选
                # 必选F强制选中，可选F按概率选择
                if f in self.mandatory_F:
                    # 必选F必须初始化配置
                    allowed_FM = self.f_fm_map.get(f, [])
                    self._initialize_fm(individual, fam, f, allowed_FM)
                else:
                    # 可选F按概率选择（40%选中率）
                    if random.random() > 0.6:
                        allowed_FM = self.f_fm_map.get(f, [])
                        self._initialize_fm(individual, fam, f, allowed_FM)
        
        individual['supplier_count'] = self.calculate_supplier_count(individual['y'])
        return individual
    
    def _initialize_fm(self, individual, fam, f, allowed_FM):
        """初始化FM层级（针对常用FM调整选择概率）"""
        for fm in allowed_FM:  # FM（基本模块）
            # 常用FM选择概率更高（70%选中率），其他FM为40%
            fm_select_prob = 0.3 if fm in self.common_FM else 0.6
            if random.random() > fm_select_prob:
                allowed_DMAM = self.fm_dmam_map.get(fm, [])
                for dmam in allowed_DMAM:  # DMAM
                    allowed_OP = self.dmam_op_map.get(dmam, [])
                    # 筛选有供应商支持的选项
                    valid_OP = [op for op in allowed_OP 
                               if any(op in self.supplier_op_map.get(s, []) for s in range(self.S))]
                    if valid_OP:
                        op = random.choice(valid_OP)
                        individual['x'][fam, f, fm, dmam, op] = 1
                        # 分配供应商
                        valid_S = [s for s in range(self.S) if op in self.supplier_op_map.get(s, [])]
                        if valid_S:
                            s = random.choice(valid_S)
                            individual['y'][f, fm, dmam, op, s] = 1
    
    def calculate_supplier_count(self, y_matrix) -> np.ndarray:
        """计算每个F的供应商数量"""
        count = np.zeros(self.F)
        for f in range(self.F):
            used_suppliers = set()
            for s in range(self.S):
                if np.sum(y_matrix[f, :, :, :, s]) > 0:
                    used_suppliers.add(s)
            count[f] = len(used_suppliers)
        return count
    
    def evaluate_individual(self, individual: Dict) -> Dict:
        """评估个体（移除产品族-F映射限制）"""
        # 1. 层级约束校验（仅保留F-FM等映射校验）
        for fam in range(self.FAMILY):  # 所有产品族均可使用所有F
            for f in range(self.F):  # 遍历所有F
                # 必选F必须保留配置，若未配置则强制初始化
                if f in self.mandatory_F and individual['x'][fam, f, :, :, :].sum() == 0:
                    allowed_FM = self.f_fm_map.get(f, [])
                    self._initialize_fm(individual, fam, f, allowed_FM)
                
                # FM层级校验（仅保留F对FM的限制）
                allowed_FM = self.f_fm_map.get(f, [])
                for fm in range(self.FM):
                    if fm not in allowed_FM:
                        individual['x'][fam, f, fm, :, :] = 0
                        individual['y'][f, fm, :, :, :] = 0
                
                # DMAM和选项层级校验
                for fm in allowed_FM:
                    allowed_DMAM = self.fm_dmam_map.get(fm, [])
                    for dmam in range(self.DMAM):
                        if dmam not in allowed_DMAM:
                            individual['x'][fam, f, fm, dmam, :] = 0
                            individual['y'][f, fm, dmam, :, :] = 0
                        
                        allowed_OP = self.dmam_op_map.get(dmam, [])
                        for op in range(self.OP):
                            if op not in allowed_OP:
                                individual['x'][fam, f, fm, dmam, op] = 0
                                individual['y'][f, fm, dmam, op, :] = 0
        
        # 2. 修复选项-供应商一致性
        for fam in range(self.FAMILY):
            for f in range(self.F):
                for fm in range(self.FM):
                    for dmam in range(self.DMAM):
                        for op in range(self.OP):
                            if individual['x'][fam, f, fm, dmam, op] > 0 and individual['y'][f, fm, dmam, op, :].sum() == 0:
                                valid_S = [s for s in range(self.S) if op in self.supplier_op_map.get(s, [])]
                                if valid_S:
                                    s = random.choice(valid_S)
                                    individual['y'][f, fm, dmam, op, s] = 1
        
        # 3. 计算总效用（按市场分别计算后汇总）
        total_utility = 0
        for market in range(self.MARKETS):  # 遍历每个市场
            market_total = 0
            for fam in range(self.FAMILY):
                # 产品族在该市场的效用
                fam_utility = 0
                for f in range(self.F):
                    for fm in range(self.FM):
                        for dmam in range(self.DMAM):
                            for op in range(self.OP):
                                fam_utility += individual['x'][fam, f, fm, dmam, op] * self.module_weight[f, fm] * self.utility[market, dmam, op]
                fam_utility += np.random.normal(0, 0.1)  # 市场噪声
                
                # MNL模型计算该产品族在市场中的选择概率
                exp_utility = np.exp(self.mnl_param * fam_utility)
                sum_exp = np.sum([np.exp(self.mnl_param * self._calc_family_utility(market, fam_j, individual)) 
                                for fam_j in range(self.FAMILY)])
                select_prob = exp_utility / (sum_exp + 1e-6)
                
                # 累加市场效用（效用×概率×市场需求）
                market_total += fam_utility * select_prob * self.market_demand[market]
            total_utility += market_total
        
        # 4. 计算总成本
        total_cost = self.calculate_total_cost(individual)
        
        # 5. 适应度（效用/成本）
        individual['fitness'] = total_utility / (total_cost + 1e-6)
        individual['total_utility'] = total_utility
        individual['total_cost'] = total_cost
        individual['supplier_count'] = self.calculate_supplier_count(individual['y'])
        
        return individual
    
    def _calc_family_utility(self, market: int, fam: int, individual: Dict) -> float:
        """计算产品族在特定市场的效用"""
        utility = 0
        for f in range(self.F):
            for fm in range(self.FM):
                for dmam in range(self.DMAM):
                    for op in range(self.OP):
                        utility += individual['x'][fam, f, fm, dmam, op] * self.module_weight[f, fm] * self.utility[market, dmam, op]
        return utility + np.random.normal(0, 0.1)
    
    def calculate_total_cost(self, individual: Dict) -> float:
        """计算总成本（制造成本+风险成本+固定成本+协调成本）"""
        # 1. 制造成本
        manufacturing_cost = 0.0
        for f in range(self.F):
            for fm in range(self.FM):
                for dmam in range(self.DMAM):
                    for op in range(self.OP):
                        for s in range(self.S):
                            if individual['y'][f, fm, dmam, op, s] > 0:
                                opid = op + 1
                                supplier_id = s + 1
                                manufacturing_cost += self.manufacturing_cost.get((supplier_id, opid), 0)
        
        # 2. 风险成本
        risk_cost = 0.0
        used_suppliers = set()
        for f in range(self.F):
            for fm in range(self.FM):
                for dmam in range(self.DMAM):
                    for op in range(self.OP):
                        for s in range(self.S):
                            if individual['y'][f, fm, dmam, op, s] > 0:
                                used_suppliers.add(s + 1)
        for s in used_suppliers:
            risk_cost += self.risk_cost.get(s, 0)
        
        # 3. 固定成本与协调成本
        fixed_cost = 1000
        coord_cost = 200 * sum(np.sqrt(individual['supplier_count'][f]) for f in range(self.F))
        
        return fixed_cost + manufacturing_cost + risk_cost + coord_cost
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉操作（基于所有F均可选用的逻辑）"""
        child = self.initialize_individual()
        
        # 产品族-F-FM层级交叉
        for fam in range(self.FAMILY):
            for f in range(self.F):  # 所有F均可参与交叉
                # 必选F交叉概率更低（保留更多父代特征）
                if f in self.mandatory_F:
                    cross_prob = 0.2  # 必选F交叉概率20%
                else:
                    cross_prob = 0.6  # 可选F交叉概率60%
                
                for fm in self.f_fm_map.get(f, []):
                    # 常用FM交叉概率更低
                    fm_cross_prob = 0.2 if fm in self.common_FM else cross_prob
                    if random.random() > fm_cross_prob:
                        child['x'][fam, f, fm, :, :] = parent1['x'][fam, f, fm, :, :]
                    else:
                        child['x'][fam, f, fm, :, :] = parent2['x'][fam, f, fm, :, :]
        
        # 供应商配置交叉
        for f in range(self.F):
            if f in self.mandatory_F:
                cross_prob = 0.2
            else:
                cross_prob = 0.6
            for fm in range(self.FM):
                fm_cross_prob = 0.2 if fm in self.common_FM else cross_prob
                for dmam in range(self.DMAM):
                    for op in range(self.OP):
                        if random.random() > fm_cross_prob:
                            child['y'][f, fm, dmam, op, :] = parent1['y'][f, fm, dmam, op, :]
                        else:
                            child['y'][f, fm, dmam, op, :] = parent2['y'][f, fm, dmam, op, :]
        
        child['supplier_count'] = self.calculate_supplier_count(child['y'])
        return child
    
    def mutate(self, individual: Dict, mutation_rate: float) -> Dict:
        """变异操作（必选F不可取消，常用FM变异率更低）"""
        mutated = {
            'x': individual['x'].copy(),
            'y': individual['y'].copy(),
            'supplier_count': individual['supplier_count'].copy(),
            'fitness': None,
            'total_utility': None,
            'total_cost': None
        }
        
        # 变异产品族-F-FM-DMAM-选项配置
        for fam in range(self.FAMILY):
            for f in range(self.F):  # 所有F均可变异
                # 必选F变异率更低，且不可取消已有选项
                if f in self.mandatory_F:
                    mut_rate = mutation_rate * 0.3  # 必选F变异率为基础的30%
                else:
                    mut_rate = mutation_rate  # 可选F使用基础变异率
                
                for fm in self.f_fm_map.get(f, []):
                    # 常用FM变异率更低
                    fm_mut_rate = mut_rate * 0.5 if fm in self.common_FM else mut_rate
                    for dmam in self.fm_dmam_map.get(fm, []):
                        if random.random() < fm_mut_rate:
                            current_op = np.where(mutated['x'][fam, f, fm, dmam, :] > 0)[0]
                            if len(current_op) > 0 and f not in self.mandatory_F:
                                # 必选F不可取消选项，仅可选F可取消
                                op = current_op[0]
                                mutated['x'][fam, f, fm, dmam, op] = 0
                                mutated['y'][f, fm, dmam, op, :] = 0
                            else:
                                # 新增选项（必选F若无选项则强制新增）
                                allowed_OP = self.dmam_op_map.get(dmam, [])
                                valid_OP = [op for op in allowed_OP 
                                           if any(op in self.supplier_op_map.get(s, []) for s in range(self.S))]
                                if valid_OP:
                                    op = random.choice(valid_OP)
                                    mutated['x'][fam, f, fm, dmam, op] = 1
                                    valid_S = [s for s in range(self.S) if op in self.supplier_op_map.get(s, [])]
                                    if valid_S:
                                        s = random.choice(valid_S)
                                        mutated['y'][f, fm, dmam, op, s] = 1
        
        # 变异供应商配置
        for f in range(self.F):
            if f in self.mandatory_F:
                mut_rate = mutation_rate * 0.3
            else:
                mut_rate = mutation_rate
            for fm in range(self.FM):
                fm_mut_rate = mut_rate * 0.5 if fm in self.common_FM else mut_rate
                for dmam in range(self.DMAM):
                    for op in range(self.OP):
                        if np.sum(mutated['x'][:, f, fm, dmam, op]) > 0 and random.random() < fm_mut_rate * 0.5:
                            valid_S = [s for s in range(self.S) if op in self.supplier_op_map.get(s, [])]
                            if valid_S:
                                mutated['y'][f, fm, dmam, op, :] = 0
                                s = random.choice(valid_S)
                                mutated['y'][f, fm, dmam, op, s] = 1
        
        mutated['supplier_count'] = self.calculate_supplier_count(mutated['y'])
        return mutated
class EnhancedDataProductOptimizer(DataProductOptimizer):
    """增强版数据产品优化器（无产品族-F映射版）"""
    def __init__(self, params: Dict):
        super().__init__(params)
        self.best_history = []
        self.convergence_gen = None
        self.current_population = None
    
    def tournament_selection(self, population: List[Dict], size: int = 3) -> List[Dict]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            contestants = random.sample(population, size)
            valid_contestants = [c for c in contestants if c['fitness'] is not None]
            winner = max(valid_contestants, key=lambda x: x['fitness']) if valid_contestants else random.choice(contestants)
            selected.append(winner.copy())
        return selected
    
    def adaptive_mutation(self, individual: Dict, gen: int, max_gen: int, base_rate: float = 0.1) -> Dict:
        """自适应变异率"""
        diversity = self.calculate_diversity(self.current_population)
        diversity_factor = 1.5 if diversity < 0.3 else 1.0
        rate = base_rate * (1 - gen/max_gen) * diversity_factor
        return self.mutate(individual, rate)
    
    def calculate_diversity(self, population: List[Dict]) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 1.0
        
        # 配置多样性
        config_diversity = 0.0
        for fam in range(self.FAMILY):
            for f in range(self.F):
                for fm in range(self.FM):
                    unique_ops = set()
                    for ind in population:
                        op = ind['x'][fam, f, fm, :, :].argmax() if ind['x'][fam, f, fm, :, :].sum() > 0 else -1
                        unique_ops.add(op)
                    config_diversity += len(unique_ops) / len(population)
        config_diversity /= (self.FAMILY * self.F * self.FM)
        
        # 适应度多样性
        fitness_scores = [ind['fitness'] for ind in population if ind['fitness'] is not None]
        if not fitness_scores:
            return 0.0
        fitness_diversity = np.std(fitness_scores) / (np.mean(fitness_scores) + 1e-6)
        
        return 0.7 * config_diversity + 0.3 * fitness_diversity
    
    def run_enhanced_ga(self, pop_size: int = 100, gens: int = 50, cx_rate: float = 0.85, 
                   base_mut_rate: float = 0.15, elite_size: int = 5, verbose: bool = True) -> Tuple[Dict, float, Dict]:
        """增强版遗传算法主流程"""
        history = {
            'best_fitness': [], 'avg_fitness': [], 'total_utility': [], 'total_cost': [],
            'suppliers': [], 'diversity': [], 'mandatory_F_usage': [], 'common_FM_usage': [],
            'market_utility': [[] for _ in range(self.MARKETS)],  # 各市场效用记录
            'convergence': None, 'iteration_time': [], 'total_time': 0.0
        }
        
        total_start = time.time()
        self.current_population = [self.initialize_individual() for _ in range(pop_size)]
        
        for gen in range(gens):
            iter_start = time.time()
            
            # 评估种群
            for ind in self.current_population:
                if ind['fitness'] is None:
                    self.evaluate_individual(ind)
            
            # 过滤无效个体
            valid_pop = [ind for ind in self.current_population if ind['fitness'] is not None]
            if not valid_pop:
                raise ValueError("所有个体评估失败，请检查参数")
            
            # 记录当前代数据
            current_best = max(valid_pop, key=lambda x: x['fitness'])
            history['best_fitness'].append(current_best['fitness'])
            history['avg_fitness'].append(np.mean([ind['fitness'] for ind in valid_pop]))
            history['total_utility'].append(np.mean([ind['total_utility'] for ind in valid_pop]))
            history['total_cost'].append(np.mean([ind['total_cost'] for ind in valid_pop]))
            history['diversity'].append(self.calculate_diversity(valid_pop))
            
            # 记录必选F和常用FM的使用率
            mandatory_F_usage = sum(ind['x'][:, f, :, :, :].sum() > 0 for ind in valid_pop for f in self.mandatory_F) / (len(valid_pop)*len(self.mandatory_F))
            history['mandatory_F_usage'].append(mandatory_F_usage)
            common_FM_usage = sum(ind['x'][:, :, fm, :, :].sum() > 0 for ind in valid_pop for fm in self.common_FM) / (len(valid_pop)*len(self.common_FM))
            history['common_FM_usage'].append(common_FM_usage)
            
            # 记录各市场效用
            for market in range(self.MARKETS):
                market_utils = []
                for ind in valid_pop:
                    utils = 0
                    for fam in range(self.FAMILY):
                        utils += self._calc_family_utility(market, fam, ind) * self.market_demand[market]
                    market_utils.append(utils)
                history['market_utility'][market].append(np.mean(market_utils))
            
            # 供应商统计
            used_suppliers = set()
            for ind in valid_pop:
                for s in range(self.S):
                    if ind['y'][:, :, :, :, s].sum() > 0:
                        used_suppliers.add(s)
            history['suppliers'].append(used_suppliers)
            
            # 检测收敛
            if gen >= 10:
                recent_improve = abs(np.mean(history['best_fitness'][-10:-5]) - np.mean(history['best_fitness'][-5:]))
                if recent_improve < 1e-3 * history['best_fitness'][-1]:
                    self.convergence_gen = gen
                    history['convergence'] = gen
                    if verbose:
                        print(f"算法在第{gen+1}代收敛")
                    break
            
            # 精英保留
            elite_size = max(2, int(pop_size * 0.05))
            elites = sorted(valid_pop, key=lambda x: x['fitness'], reverse=True)[:elite_size]
            
            # 选择与交叉
            selected = self.tournament_selection(valid_pop)
            offspring = []
            for i in range(0, len(selected)-1, 2):
                p1, p2 = selected[i], selected[i+1]
                if random.random() < cx_rate:
                    offspring.append(self.crossover(p1, p2))
                    offspring.append(self.crossover(p2, p1))
                else:
                    offspring.append(p1.copy())
                    offspring.append(p2.copy())
            
            # 变异
            for i in range(len(offspring)):
                if random.random() < base_mut_rate:
                    offspring[i] = self.adaptive_mutation(offspring[i], gen, gens, base_mut_rate)
            
            # 评估子代
            for ind in offspring:
                self.evaluate_individual(ind)
            
            # 形成新一代种群
            self.current_population = elites + offspring[:pop_size - elite_size]
            
            # 打印进度
            if verbose and (gen % 5 == 0 or gen == gens-1):
                avg_time = np.mean(history['iteration_time'][-5:]) if gen >=5 else (time.time() - iter_start)
                remaining_time = avg_time * (gens - gen - 1)
                self._print_progress(gen+1, gens, current_best, history, len(used_suppliers), time.time()-iter_start, remaining_time)
            
            history['iteration_time'].append(time.time() - iter_start)
        
        # 最终统计
        history['total_time'] = time.time() - total_start
        history['convergence'] = self.convergence_gen if self.convergence_gen is not None else gens-1
        best_ind = max(self.current_population, key=lambda x: x['fitness'])
        
        print(f"\n优化完成：")
        print(f"最佳适应度: {best_ind['fitness']:.4f} | 总效用: {best_ind['total_utility']:.2f} | 总成本: {best_ind['total_cost']:.2f}")
        return best_ind, best_ind['fitness'], history
    
    def _print_progress(self, gen: int, max_gen: int, best: Dict, history: Dict, supplier_count: int, iter_time: float, remaining_time: float):
        """打印进度"""
        print(
            f"代 {gen:3d}/{max_gen} | "
            f"最佳适应度: {best['fitness']:.4f} | "
            f"平均适应度: {history['avg_fitness'][-1]:.4f} | "
            f"必选F使用率: {history['mandatory_F_usage'][-1]:.2f} | "
            f"常用FM使用率: {history['common_FM_usage'][-1]:.2f} | "
            f"供应商: {supplier_count}个 | "
            f"耗时: {iter_time:.2f}s | 剩余: {remaining_time:.2f}s"
        )
    
    def print_config_detail(self, solution: Dict):
        """打印配置详情（不区分产品族对F的限制）"""
        family_names = {i: f"产品族{i+1}" for i in range(self.FAMILY)}
        f_names = {i: f"F{i+1}（复合模块，{'必选' if i in self.mandatory_F else '可选'}）" for i in range(self.F)}
        fm_names = {i: f"FM{i+1}（基本模块，{'常用' if i in self.common_FM else '可选'}）" for i in range(self.FM)}
        dmam_names = {i: f"DMAM{i+1}（数据模块）" for i in range(self.DMAM)}
        op_names = {i: f"选项{i+1}" for i in range(self.OP)}
        supplier_names = {
            0: "Google", 1: "阿里云PAI", 2: "华为ModelArts", 3: "Microsoft", 4: "腾讯云",
            5: "蚂蚁OceanBase", 6: "Apache Flink", 7: "DataRobot", 8: "NVIDIA", 9: "Hugging Face",
            10: "国家企业信用信息公示系统", 11: "中国人民银行征信中心", 12: "中国裁判文书网",
            13: "信用中国", 14: "人力资源和社会保障部", 15: "自然资源部",
            16: "地方登记中心", 17: "各商业银行", 18: "四大会计师事务所", 19: "国家级行业协会",
            20: "行业会员企业", 21: "商业信息平台", 22: "市场调研机构", 23: "法律数据库",
            24: "商业征信平台", 25: "官方媒体", 26: "自媒体平台", 27: "企业ERP系统",
            28: "电商平台", 29: "国家统计局"
        }
        
        print("\n产品配置详情：")
        for fam in range(self.FAMILY):
            print(f"\n{family_names[fam]}:")
            for f in range(self.F):  # 所有F均可被产品族选用
                if solution['x'][fam, f, :, :, :].sum() == 0:
                    continue
                print(f"  {f_names[f]}:")
                for fm in self.f_fm_map.get(f, []):
                    if solution['x'][fam, f, fm, :, :].sum() == 0:
                        continue
                    print(f"    {fm_names[fm]}:")
                    for dmam in self.fm_dmam_map.get(fm, []):
                        if solution['x'][fam, f, fm, dmam, :].sum() == 0:
                            continue
                        print(f"      {dmam_names[dmam]}:")
                        for op in self.dmam_op_map.get(dmam, []):
                            if solution['x'][fam, f, fm, dmam, op] > 0:
                                suppliers = [supplier_names[s] for s in range(self.S) if solution['y'][f, fm, dmam, op, s] > 0]
                                print(f"        {op_names[op]}（供应商：{', '.join(suppliers)}）")
def visualize_results(history: Dict, save_path: str = 'ga_results.png'):
    """可视化结果"""
    plt.figure(figsize=(18, 18))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    gens = len(history['best_fitness'])
    
    # 确保迭代时间数据与代数一致
    if len(history['iteration_time']) < gens:
        history['iteration_time'].append(0)  # 补充最后一个代的时间
    
    # 1. 适应度趋势
    plt.subplot(4, 2, 1)
    plt.plot(range(gens), history['best_fitness'], 'b-', label='最佳适应度', linewidth=2)
    plt.plot(range(gens), history['avg_fitness'], 'g--', label='平均适应度')
    if history['convergence'] is not None:
        plt.axvline(history['convergence'], color='r', linestyle=':', label=f'收敛点（代{history["convergence"]+1}）')
    plt.xlabel('遗传代数')
    plt.ylabel('适应度（效用/成本）')
    plt.title('适应度进化趋势')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. 效用-成本分布
    plt.subplot(4, 2, 2)
    utils = history['total_utility']
    costs = history['total_cost']
    plt.scatter(utils, costs, c=range(gens), cmap='viridis', alpha=0.6)
    plt.colorbar(label='代数')
    plt.xlabel('平均总效用')
    plt.ylabel('平均总成本')
    plt.title('效用-成本分布关系')
    plt.grid(alpha=0.3)
    
    # 3. 种群多样性与必选/常用模块使用率
    plt.subplot(4, 2, 3)
    plt.plot(range(gens), history['diversity'], 'purple', label='种群多样性', linewidth=2)
    plt.plot(range(gens), history['mandatory_F_usage'], 'red', label='必选F平均使用率', linewidth=2)
    plt.plot(range(gens), history['common_FM_usage'], 'orange', label='常用FM平均使用率', linewidth=2)
    plt.axhline(0.3, color='r', linestyle='--', label='多样性阈值')
    plt.xlabel('遗传代数')
    plt.ylabel('指数/率')
    plt.title('种群多样性与模块使用率')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. 各市场效用对比
    plt.subplot(4, 2, 4)
    for market in range(len(history['market_utility'])):
        plt.plot(range(gens), history['market_utility'][market], label=f'市场{market+1}')
    plt.xlabel('遗传代数')
    plt.ylabel('平均效用')
    plt.title('各市场效用趋势对比')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. 必选F与常用FM使用率单独对比
    plt.subplot(4, 2, 5)
    plt.plot(range(gens), history['mandatory_F_usage'], 'red', label='必选F使用率', linewidth=2)
    plt.plot(range(gens), history['common_FM_usage'], 'orange', label='常用FM使用率', linewidth=2)
    plt.axhline(1.0, color='gray', linestyle='--', label='完全使用率')
    plt.xlabel('遗传代数')
    plt.ylabel('使用率')
    plt.title('必选F与常用FM使用率趋势')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. 计算时间分析
    plt.subplot(4, 2, 6)
    plt.plot(range(gens), history['iteration_time'], 'm-', label='每代计算时间（s）')
    if gens > 1:
        plt.axhline(np.mean(history['iteration_time']), color='b', linestyle='--', label='平均耗时')
    plt.xlabel('遗传代数')
    plt.ylabel('时间（s）')
    plt.title('计算效率分析')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        print(f"可视化结果已保存至 {save_path}")
    except Exception as e:
        print(f"保存图片失败：{e}")
    plt.show()
def save_results(best_solution: Dict, history: Dict, params: Dict, filename: str = 'optimization_results.json'):
    """保存结果为JSON"""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    data = {
        'parameters': params,
        'best_solution': {
            'x': convert_numpy(best_solution['x']),
            'y': convert_numpy(best_solution['y']),
            'supplier_count': convert_numpy(best_solution['supplier_count']),
            'fitness': best_solution['fitness'],
            'total_utility': best_solution['total_utility'],
            'total_cost': best_solution['total_cost']
        },
        'history': {
            'best_fitness': history['best_fitness'],
            'avg_fitness': history['avg_fitness'],
            'total_utility': history['total_utility'],
            'total_cost': history['total_cost'],
            'diversity': history['diversity'],
            'mandatory_F_usage': history['mandatory_F_usage'],
            'common_FM_usage': history['common_FM_usage'],
            'market_utility': history['market_utility'],
            'convergence': history['convergence'],
            'total_time': history['total_time']
        }
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=convert_numpy)
        print(f"优化结果已保存至 {filename}")
    except Exception as e:
        print(f"保存JSON失败：{e}")
if __name__ == "__main__":
    # --------------------------
    # 1. 层级映射参数（移除family_f_map）
    # --------------------------
    
    # 1.2 F→FM映射
    f_fm_map = {
        0: [2, 3, 6, 8],   # F1包含FM3,FM4,FM7,FM9 (索引2,3,6,8)
        1: [2, 4, 7, 9],   # F2包含FM3,FM5,FM8,FM10 (索引2,4,7,9)
        2: [1, 2, 3, 5],   # F3（必选）包含FM2,FM3,FM4,FM6 (索引1,2,3,5)
        3: [2, 3, 6, 8],   # F4包含FM3,FM4,FM7,FM9 (索引2,3,6,8)
        4: [1, 5, 10, 11], # F5包含FM2,FM6,FM11,FM12 (索引1,5,10,11)
        5: [0, 2, 3, 4],   # F6（必选）包含FM1,FM3,FM4,FM5 (索引0,2,3,4)
        6: [5, 7, 9, 10],  # F7包含FM6,FM8,FM10,FM11 (索引5,7,9,10)
        7: [4, 5, 7, 10]   # F8（必选）包含FM5,FM6,FM8,FM11 (索引4,5,7,10)
    }
    
    # 1.3 FM→DMAM映射
    fm_dmam_map = {
        0: [0, 1, 2,9],   
        1: [1, 11],    
        2: [0, 6, 13, 14],
        3: [3, 4, 15],  
        4: [0, 2, 5, 16], 
        5: [2, 4, 10],  
        6: [5, 12],
        7: [4, 6, 10, 17],
        8: [7, 10, 18], 
        9: [6, 10, 19], 
        10: [4, 10],
        11: [4, 8,9, 20] 
    }
    
    # 1.4 DMAM→选项映射
    dmam_op_map = {
        0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11],
        6: [12, 13], 7: [14, 15], 8: [16, 17], 9: [18, 19], 10: [20, 21],
        11: [22, 23], 12: [24, 25], 13: [26, 27], 14: [28, 29], 15: [30, 31],
        16: [32, 33], 17: [34, 35], 18: [36, 37], 19: [38, 39], 20: [40, 41]
    }
    
    # 1.5 供应商→选项映射
    supplier_op_map = {
        0: [0, 8], 1: [1, 5,16], 2: [4, 9], 3: [2, 13, 14], 4: [7, 11], 5: [15],
        6: [6, 17], 7: [10], 8: [3], 9: [12], 10: [18], 11: [30, 35], 12: [26],
        13: [28], 14: [40], 15: [38], 16: [39], 17: [34], 18: [21], 19: [37],
        20: [36], 21: [19, 29], 22: [23], 23: [27], 24: [31], 25: [32], 26: [33],
        27: [20, 24, 41], 28: [25], 29: [22]
    }
    
    # --------------------------
    # 2. 成本数据
    # --------------------------
    manufacturing_cost = {
        (1, 1): 500, (1, 9): 150, 
        (2, 2): 300, (2, 6): 350, (2,17):300,
        (3, 5): 200, (3, 10): 250,
        (4, 3): 400, (4, 14): 500, (4, 15): 200, 
        (5, 8): 300, (5, 12): 450, 
        (6, 16): 150,
        (7, 7): 100, (7, 18): 350, 
        (8, 11): 400, 
        (9, 4): 600, 
        (10, 13): 700, 
        (11, 19): 80,
        (12, 31): 600, (12, 36): 600,
        (13, 27): 50,
        (14, 29): 150, 
        (15, 41): 200, 
        (16, 39): 300,
        (17, 40): 150,
        (18, 35): 400, 
        (19, 22): 400,
        (20, 38): 200, 
        (21, 37): 100,
        (22, 20): 100,(22, 30): 100,
        (23, 24): 300,
        (24, 28): 150, 
        (25, 32): 300,
        (26, 33): 50,
        (27, 34): 80,
        (28, 21): 200, (28, 25): 100, (28, 42): 100, 
        (29, 26): 150,
        (30, 23): 100
    }
    
    risk_cost = {
        1: 50, 2: 30, 3: 20, 4: 40, 5: 25, 6: 15, 7: 10, 8: 60, 9: 80, 10: 20,
        11: 1, 12: 5, 13: 1, 14: 2, 15: 3, 16: 5, 17: 10, 18: 50, 19: 30, 20: 10,
        21: 30, 22: 25, 23: 30, 24: 20, 25: 40, 26: 5, 27: 50, 28: 20, 29: 40, 30: 1
    }
    
    # --------------------------
    # 3. 运行优化器
    # --------------------------
    params = {
        'markets': 5,  # 市场数量
        'product_families': 4,
        'composite_modules': 8,
        'basic_modules': 12,
        'tech_info_modules': 21,
        'options': 42,
        'suppliers': 30,
        'mu': 1.0,
        # 移除family_f_map参数
        'f_fm_map': f_fm_map,
        'fm_dmam_map': fm_dmam_map,
        'dmam_op_map': dmam_op_map,
        'supplier_op_map': supplier_op_map,
        'manufacturing_cost': manufacturing_cost,
        'risk_cost': risk_cost
    }
    
    optimizer = EnhancedDataProductOptimizer(params)
    
    # 运行优化
    print("启动增强版遗传算法优化（无产品族-复合模块映射）...")
    best_solution, best_fitness, history = optimizer.run_enhanced_ga(
        pop_size=70,
        gens=100,
        cx_rate=0.8,
        base_mut_rate=0.4,
        elite_size=5,
        verbose=True
    )
    
    # 结果分析
    print(f"\n优化完成，总耗时 {history['total_time']:.2f}秒")
    print(f"平均每代耗时: {np.mean(history['iteration_time']):.4f}s")
    print(f"最佳解决方案适应度: {best_fitness:.4f}")
    print(f"使用供应商数量: {len([s for s in range(params['suppliers']) if best_solution['y'][:,:,:,:,s].sum() > 0])}")
    print(f"算法收敛于第 {history['convergence']+1}代")
    
    # 输出配置详情
    optimizer.print_config_detail(best_solution)
    
    # 可视化与保存结果
    visualize_results(history)
    save_results(best_solution, history, params)