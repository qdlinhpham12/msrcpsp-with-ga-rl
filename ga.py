from deap import base, creator, tools
import random
from typing import List, Tuple, Dict
from data import parse_imopse_file  # Nhập module data để lấy tasks và resources

# Định nghĩa các loại fitness và cá thể (minimize makespan)
# Sử dụng type hints để giảm warning trong PyCharm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize makespan
creator.create("Individual", list, fitness=creator.FitnessMin)


# Hàm kiểm tra tính khả thi của thứ tự task (tuân thủ precedence)
def is_valid_order(order: List[int], tasks: Dict[int, dict]) -> bool:
    for i, task in enumerate(order):
        for pred in tasks.get(task, {}).get("predecessors", []):
            if pred not in order[:i]:
                return False
    return True


# Hàm kiểm tra xem order có chứa tất cả task từ 1 đến len(tasks) không
def is_complete_order(order: List[int], tasks: Dict[int, dict]) -> bool:
    expected_tasks = set(range(1, len(tasks) + 1))
    actual_tasks = set(order)
    return expected_tasks == actual_tasks


# Hàm heuristic initialization (topological sort)
def heuristic_init(tasks: Dict[int, dict]) -> List[int]:
    def topological_sort():
        visited = set()
        result = []

        def dfs(node: int):
            if node in visited:
                return
            visited.add(node)
            for t in tasks.get(node, {}).get("predecessors", []):
                dfs(t)
            result.append(node)

        for t in sorted(tasks.keys()):  # Đảm bảo tất cả task được duyệt
            if t not in visited and t in tasks:  # Kiểm tra task tồn tại trong dictionary
                dfs(t)
        return result

    order = topological_sort()
    while not (is_valid_order(order, tasks) and is_complete_order(order, tasks)):
        random.shuffle(order)  # Đảm bảo thứ tự hợp lệ và đầy đủ
    return creator.Individual(order)  # Trả về cá thể với fitness


# Hàm random initialization
def random_init(tasks: Dict[int, dict]) -> List[int]:
    order = list(range(1, len(tasks) + 1))  # Đảm bảo lấy tất cả task từ 1 đến 100
    random.shuffle(order)
    while not (is_valid_order(order, tasks) and is_complete_order(order, tasks)):
        random.shuffle(order)  # Sửa lại nếu vi phạm precedence hoặc không đầy đủ
    return creator.Individual(order)  # Trả về cá thể với fitness


# Hàm tạo cá thể (chromosome: chỉ thứ tự task)
def create_individual(tasks: Dict[int, dict]) -> List[int]:
    if random.random() < 0.7:  # 70% heuristic, 30% random
        return heuristic_init(tasks)
    return random_init(tasks)


# Hàm kiểm tra ràng buộc kỹ năng và tài nguyên
def can_assign(task_id: int, resource_id: int, tasks: Dict[int, dict], resources: Dict[int, dict]) -> bool:
    task_skills = tasks.get(task_id, {}).get("skills", {})
    res_skills = resources.get(resource_id, {}).get("skills", {})
    return all(skill in res_skills and res_skills[skill] >= level for skill, level in task_skills.items())


# Hàm tính makespan (giải mã với phân bổ tài nguyên)
def evaluate_schedule(individual: List[int], tasks: Dict[int, dict], resources: Dict[int, dict]) -> float:
    if not individual or len(individual) != len(tasks) or not is_complete_order(individual, tasks):
        return float("inf")

    time = 0
    resource_usage = {res: 0 for res in resources}  # Thời điểm tài nguyên rảnh
    completed = set()

    for task_id in individual:
        if task_id not in tasks:  # Kiểm tra task tồn tại
            return float("inf")

        # Chọn resource khả thi ngẫu nhiên
        available_resources = [r for r in resources if can_assign(task_id, r, tasks, resources)]
        if not available_resources:
            return float("inf")  # Không có resource phù hợp, phạt

        # Chọn resource rảnh sớm nhất
        resource_id = min(available_resources, key=lambda r: resource_usage[r])

        # Chờ predecessors hoàn thành
        start_time = max([resource_usage[resource_id]] +
                         [resource_usage[r] for r in resources if
                          any(t in completed for t in tasks[task_id]["predecessors"])],
                         default=0)

        end_time = start_time + tasks[task_id]["duration"]
        resource_usage[resource_id] = end_time
        completed.add(task_id)
        time = max(time, end_time)

    if len(completed) != len(tasks):
        return float("inf")  # Vi phạm, phạt
    return time  # Makespan


# Hàm repair (sửa cá thể vi phạm ràng buộc)
def repair_individual(individual: List[int], tasks: Dict[int, dict]) -> List[int]:
    valid_order = heuristic_init(tasks)  # Sử dụng topological sort để sửa
    return creator.Individual(valid_order)  # Trả về cá thể với fitness


# Cấu hình GA
def setup_ga(tasks: Dict[int, dict], resources: Dict[int, dict]) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(tasks))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (evaluate_schedule(ind, tasks, resources),))

    # Sử dụng cxOrdered an toàn hơn, kiểm tra độ dài và hợp lệ của cá thể
    def safe_cxOrdered(ind1: List[int], ind2: List[int]) -> Tuple[List[int], List[int]]:
        if len(ind1) != len(ind2) or len(ind1) != len(tasks) or not is_complete_order(ind1,
                                                                                      tasks) or not is_complete_order(
                ind2, tasks):
            return creator.Individual(ind1[:]), creator.Individual(ind2[:])  # Trả về cá thể hợp lệ nếu không hợp lệ
        try:
            child1, child2 = tools.cxOrdered(ind1, ind2)
            # Đảm bảo trả về cá thể với fitness
            return creator.Individual(child1), creator.Individual(child2)
        except IndexError:
            return repair_individual(ind1, tasks), repair_individual(ind2, tasks)

    toolbox.register("mate", safe_cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Swap mutation
    toolbox.register("select", tools.selTournament, tournsize=5)

    return toolbox


# Hàm chạy GA
def run_ga(tasks: Dict[int, dict], resources: Dict[int, dict], pop_size: int = 100, n_gen: int = 50000) -> Tuple[
    List[int], float]:
    toolbox = setup_ga(tasks, resources)
    pop = toolbox.population(n=pop_size)

    # Adaptive mutation rate
    mutation_rate = 0.05
    best_makespan = float("inf")

    for gen in range(n_gen+1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Xác suất lai ghép cao
                child1, child2 = toolbox.mate(child1, child2)
                # Kiểm tra và sửa nếu vi phạm ràng buộc sau lai ghép
                if not is_valid_order(child1, tasks) or not is_complete_order(child1, tasks):
                    child1 = repair_individual(child1, tasks)
                if not is_valid_order(child2, tasks) or not is_complete_order(child2, tasks):
                    child2 = repair_individual(child2, tasks)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                # Kiểm tra và sửa nếu vi phạm ràng buộc sau đột biến
                if not is_valid_order(mutant, tasks) or not is_complete_order(mutant, tasks):
                    mutant = repair_individual(mutant, tasks)
                del mutant.fitness.values  # Xóa fitness để đánh giá lại

        # Đánh giá lại fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = offspring

        # Cập nhật mutation rate (adaptive mutation)
        current_best = min(pop, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        if current_best < best_makespan:
            best_makespan = current_best
            mutation_rate *= 0.9  # Giảm mutation rate nếu tiến bộ
        else:
            mutation_rate = min(mutation_rate * 1.1, 0.2)  # Tăng mutation rate nếu hội tụ
        if gen % 5000 == 0:
            print(f"Generation {gen}: Best makespan = {best_makespan}")

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_makespan


# Ví dụ sử dụng
"""if __name__ == "__main__":
    # Tải dữ liệu từ module data
    tasks, resources = parse_imopse_file("D:/NCKHSV/imopse_validator_pack/IMOPSE/def_small/10_5_8_5.def")
    # Đảm bảo tasks chỉ chứa các TaskID liên tục từ 1 đến 100
    sorted_tasks = {k: v for k, v in sorted(tasks.items()) if 1 <= k <= 100}
    best_individual, best_makespan = run_ga(sorted_tasks, resources)
    print("Best schedule (task order):", best_individual)
    print("Best makespan:", best_makespan)"""