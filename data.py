import re


def parse_imopse_file(file_path):
    """
    Parse data from iMOPSE .def file and return tasks and resources dictionaries.

    Args:
        file_path (str): Path to the .def file.

    Returns:
        tuple: (tasks, resources) dictionaries.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"File không tồn tại: {file_path}")
        return None, None

    tasks = {}
    resources = {}
    task_section = False
    resource_section = False

    lines = content.split('\n')
    for line in lines:
        if "TaskID" in line:
            task_section = True
            resource_section = False
            continue
        if "ResourceID" in line:
            task_section = False
            resource_section = True
            continue

        line = line.strip()
        if not line or line.startswith("---") or line.startswith("General characteristics:"):
            continue

        if task_section and "TaskID" not in line:
            match = re.match(r"(\d+)\s+(\d+)\s+(Q\d+:\s*\d+)\s*(.*)", line)
            if match:
                task_id, duration, skills, preds = match.groups()
                task_id = int(task_id)
                duration = int(duration)
                # Chỉ lấy skill đầu tiên cho mỗi task (mỗi task có 1 skill duy nhất)
                if skills:
                    skill_name, skill_level = skills.split(':')
                    skills_dict = {skill_name: int(skill_level)}
                else:
                    skills_dict = {}  # Trường hợp không có skill
                preds = [int(p) for p in preds.split() if p and p.strip().isdigit()] if preds else []
                tasks[task_id] = {"duration": duration, "skills": skills_dict, "predecessors": preds}

        if resource_section and "ResourceID" not in line:
            match = re.match(r"(\d+)\s+(\d+\.\d+)\s*(.*)", line)
            if match:
                resource_id, salary, skills_str = match.groups()
                resource_id = int(resource_id)
                skills = {}
                # Parse tất cả các cặp Q#: level cho resource, tích lũy thay vì ghi đè
                if skills_str:
                    # Lấy tất cả các cặp Q#: level bằng regex
                    skills_pairs = [s.strip() for s in re.findall(r"Q\d+:\s*\d+", skills_str) if s.strip()]
                    for s in skills_pairs:
                        try:
                            skill_name, skill_level = s.split(':')
                            # Loại bỏ khoảng trắng trong skill_level
                            skill_level = skill_level.strip()
                            # Tích lũy tất cả các skill vào dictionary
                            skills.update({skill_name:int(skill_level)}) if skill_level else 0
                        except (ValueError, IndexError):
                            continue  # Bỏ qua nếu không parse được
                resources[resource_id] = {"salary": float(salary), "skills": skills}

    return tasks, resources


# Hàm main kiểm thử code
if __name__ == "__main__":
    tasks, resources = parse_imopse_file("D:/NCKHSV/imopse_validator_pack/IMOPSE/def_small/10_3_5_3.def")
    print("Tasks:", len(tasks))
    print("Resources:", len(resources))
    n = int(input("Nhập TaskID cần kiểm tra: "))
    print("Sample Task:", tasks[n])  # Hiển thị mẫu Task 30
    m = int(input("Nhập ResourceID cần kiểm tra: "))
    print("Sample Resource:", resources[m]) if m <= 3 else 0 # Hiển thị mẫu Resource 5
