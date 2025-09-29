import copy
import sys

# 부동소수점 비교를 위한 작은 값
EPSILON = 1e-9

def get_matrix_input():
    """사용자로부터 n x n 행렬을 입력받아 2차원 리스트로 반환하는 함수"""
    while True:
        try:
            n = int(input("행렬의 크기 n을 입력하세요: "))
            if n <= 0:
                print("n은 0보다 큰 정수여야 합니다.")
                continue
            break
        except ValueError:
            print("유효한 정수를 입력하세요.")

    matrix = []
    print(f"{n}x{n} 행렬의 원소를 행 단위로 입력하세요 (각 원소는 공백으로 구분).")
    for i in range(n):
        while True:
            try:
                row_input = input(f"{i+1}번째 행 입력: ").split()
                if len(row_input) != n:
                    print(f"정확히 {n}개의 원소를 입력해야 합니다.")
                    continue
                row = [float(x) for x in row_input]
                matrix.append(row)
                break
            except ValueError:
                print("유효한 숫자를 입력하세요.")
    return n, matrix

def determinant(matrix):
    """재귀를 이용한 행렬식 계산 함수"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    
    det = 0
    for j in range(n):
        sign = (-1) ** j
        minor = get_minor(matrix, 0, j)
        det += sign * matrix[0][j] * determinant(minor)
    return det

def get_minor(matrix, i, j):
    """(i, j) 원소의 소행렬(minor matrix)을 반환하는 함수"""
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def gaussian_elimination_inverse(matrix):
    """가우스 소거법을 이용한 역행렬 계산 함수 (추가 기능)"""
    n = len(matrix)
    
    if abs(determinant(matrix)) < EPSILON:
        return None

    mat = copy.deepcopy(matrix)
    identity = [[float(i == j) for i in range(n)] for j in range(n)]
    augmented_matrix = [mat[i] + identity[i] for i in range(n)]

    # 전진 소거
    for i in range(n):
        if abs(augmented_matrix[i][i]) < EPSILON:
            for k in range(i + 1, n):
                if abs(augmented_matrix[k][i]) > EPSILON:
                    augmented_matrix[i], augmented_matrix[k] = augmented_matrix[k], augmented_matrix[i]
                    break
            else:
                return None
        
        pivot = augmented_matrix[i][i]
        for j in range(i, 2 * n):
            augmented_matrix[i][j] /= pivot
        
        for k in range(i + 1, n):
            factor = augmented_matrix[k][i]
            for j in range(i, 2 * n):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # 후진 대입
    for i in range(n - 1, -1, -1):
        for k in range(i - 1, -1, -1):
            factor = augmented_matrix[k][i]
            for j in range(i, 2 * n):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix

def gauss_jordan_elimination_inverse(matrix):
    """가우스-조던 소거법을 이용한 역행렬 계산 함수"""
    n = len(matrix)

    if abs(determinant(matrix)) < EPSILON:
        return None

    mat = copy.deepcopy(matrix)
    identity = [[float(i == j) for i in range(n)] for j in range(n)]
    augmented_matrix = [mat[i] + identity[i] for i in range(n)]

    for i in range(n):
        if abs(augmented_matrix[i][i]) < EPSILON:
            for k in range(i + 1, n):
                if abs(augmented_matrix[k][i]) > EPSILON:
                    augmented_matrix[i], augmented_matrix[k] = augmented_matrix[k], augmented_matrix[i]
                    break
            else:
                return None

        pivot = augmented_matrix[i][i]
        for j in range(i, 2 * n):
            augmented_matrix[i][j] /= pivot

        for k in range(n):
            if i == k:
                continue
            factor = augmented_matrix[k][i]
            for j in range(i, 2 * n):
                augmented_matrix[k][j] -= factor * augmented_matrix[i][j]
    
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix

def adjugate_method_inverse(matrix):
    """수반행렬(Adjugate Matrix)을 이용한 역행렬 계산 함수"""
    n = len(matrix)
    det = determinant(matrix)

    if abs(det) < EPSILON:
        return None
        
    cofactor_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = get_minor(matrix, i, j)
            sign = (-1) ** (i + j)
            cofactor_matrix[i][j] = sign * determinant(minor)
            
    adjugate_matrix = [[cofactor_matrix[j][i] for j in range(n)] for i in range(n)]
    inverse_matrix = [[elem / det for elem in row] for row in adjugate_matrix]
    return inverse_matrix

def compare_matrices(m1, m2):
    """두 행렬이 동일한지 비교하는 함수 (부동소수점 오차 감안)"""
    if m1 is None or m2 is None:
        return m1 is None and m2 is None
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return False
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            if abs(m1[i][j] - m2[i][j]) > EPSILON:
                return False
    return True

def print_matrix_single(matrix, name="Matrix"):
    """하나의 행렬을 보기 좋게 출력하는 함수 (주로 입력 행렬 출력에 사용)"""
    if matrix is None:
        print(f"{name} is None (e.g., non-invertible)")
        return
    print(f"--- {name} ---")
    for row in matrix:
        print(" ".join(f"{elem:9.4f}" for elem in row))
    print("-" * (len(name) + 6))

def main():
    """메인 실행 함수"""
    n, matrix = get_matrix_input()
    
    # 세 가지 방법으로 역행렬 계산
    results = {
        "가우스 소거법": gaussian_elimination_inverse(matrix),
        "가우스-조던": gauss_jordan_elimination_inverse(matrix),
        "수반행렬": adjugate_method_inverse(matrix)
    }
    
    # 1. 결과 출력 (가로 배치 - 수정된 버전)
    print("\n--- 역행렬 계산 결과 ---")
    
    valid_results = {name: mat for name, mat in results.items() if mat is not None}

    if not valid_results:
        print("역행렬이 존재하지 않습니다 (행렬식=0).")
    else:
        names = list(valid_results.keys())
        matrices = list(valid_results.values())
        
        # 각 행렬의 출력을 위한 전체 너비 계산
        # (원소 너비 10 + 공백 1) * n개 = 11*n. 넉넉하게 패딩을 줌.
        col_width = n * 11 + (n - 1)
        
        # 제목 출력
        header_line = []
        for name in names:
            # 한글/영문 너비 차이를 고려하여 패딩을 넉넉하게 줌
            header_line.append(f"{name: <{col_width}}")
        print('\t'.join(header_line))
        
        # 구분선 출력
        separator_line = []
        for _ in names:
            separator_line.append("-" * col_width)
        print('\t'.join(separator_line))
        
        # 행렬 내용 출력 (한 줄씩)
        for i in range(n):
            row_line = []
            for mat in matrices:
                row_str = " ".join(f"{elem:10.4f}" for elem in mat[i])
                row_line.append(f"{row_str: <{col_width}}")
            print('\t'.join(row_line))
            
        print('\t'.join(separator_line))

    # 2. 결과 비교 (모든 결과 비교)
    print("\n--- 결과 비교 ---")
    if len(valid_results) < 2:
        print(">> 비교할 수 있는 역행렬 결과가 충분하지 않습니다.")
    else:
        names = list(valid_results.keys())
        matrices = list(valid_results.values())
        
        all_same = True
        # 기준(첫 번째) 행렬과 나머지 행렬들을 비교
        for i in range(1, len(matrices)):
            if not compare_matrices(matrices[0], matrices[i]):
                print(f">> '{names[0]}'과(와) '{names[i]}'의 결과가 다릅니다.")
                all_same = False
        
        if all_same:
            print(">> 계산된 모든 역행렬의 결과가 동일합니다.")

if __name__ == "__main__":
    main()
