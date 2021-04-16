import random
import numpy as np
import copy
import matplotlib.pyplot as plt

N_SIZE = 5

# 1. Chromosome design
# 1~5 중 임의의 수를 배열의 각 요소에 채워넣어 배열을 생성한다.
def design_chromosome(n_size): 
    return np.random.randint(1,N_SIZE+1, size = n_size)

# 원하는 갯수의 chromosome을 만든다. - 0 세대 
def initialization_chromosome(SIZE):
    multi_chromosome = [design_chromosome(N_SIZE) for _ in range(SIZE)]
    return multi_chromosome

# Fitness evaluation
def fitness_evaluation(chromosome):  #chromosome -> 평가할 배열

    # Chromosome의 갯수 
    length = len(chromosome)
    num_of_queen = len(chromosome)

    # 1. 퀸의 갯수는 5개에서 시작한다.
    # 2. Chromosome[i] 위치에 있는 퀸의 생존여부를 검사한다.
    # 3. 만약에 같은 행에 존재하는 퀸이 있다면 해당 퀸은 공격 당하므로 사용 불가 -> 퀸 갯수 감소 후 다음 퀸으로 이동
    # 4. 같은 행에 존재하는 퀸이 없다면 대각선 상에 퀸이 존재하는지 검사한다.
    # 5. 대각선 상에 다른 퀸이 존재한다면 해당 퀸은 공격 당하므로 사용 불가 -> 퀸 갯소 감소 후 다음 퀸으로 이동
    
    for i in range(length):
        # 같은 행에 퀸이 존재하는지 확인한다.( 같은 값이 배열이 존재한다면 같은 행에 있는 것이므로 1씩 더해준다. )
        overlap = (chromosome == chromosome[i]).sum()
        if overlap > 1 :                    # 같은 행에 퀸이 중복되면 퀸의 갯수는 1보다 커지고, 이 경우 해당 퀸은 죽는 것으로 판단  
            num_of_queen -= 1               # 퀸 갯수 감소
        else:                               # 대각선 상에 겹치는 퀸이 있는 경우를 검사한다.
            for j in range(length):         # 같은 대각선 상에 있는 경우 두 개의 점 (x1,y1),(x2,y2) x축 y축의 거리차는 같음을 이용  
                if (i!=j):                  # |(x2-x1)| = |(y2-y1)| ==> 퀸의 갯수 1 감소 후 다음 퀸으로 이동
                    overlap_queen = abs(chromosome[i] - chromosome[j])
                    if( abs(i-j) == overlap_queen):
                        num_of_queen -= 1
                        break
                    
    # 살아남은 퀸의 갯수를 반환한다.
    return num_of_queen

# 점수가 매겨진 chromosome들 중에서 선택하는 함수
def selection_pick(multi_chromosome, fitness_score):
    # 적합도 점수에 대한 배열과 유전자 정보를 가지고 있는 배열을 zip을 통해합친다.
    # 적합도 점수에 따라 리스트를 정렬한다.
    # 살아남은 퀸의 갯수를 기준으로 설정하였으므로, 내림차순 정렬을 위해 reverse해준다.
    ranking_chromosome = list(zip(fitness_score, multi_chromosome))
    ranking_chromosome.sort(key = lambda x:x[0])
    ranking_chromosome.reverse()
    
    ranking_chromosome = [ i for i in ranking_chromosome if i[0] != 0]
    # 적합도 점수가 0인 유전자들을 배열에서 없애준다.

    fitness_score, multi_chromosome = zip(*ranking_chromosome)#*
    # 리스트를 적합도 점수에 대한 배열과 유전자 정보를 가지고 있는 배열로 나누어준다.
    # 0을 제외한 유전자가 10개 이상일 경우 상위 10개를 선택하고, 10개 미만일 경우 그대로 반환한다.
    if (len(multi_chromosome)>10):
        return multi_chromosome[0:10]
    else :
        return multi_chromosome

# 부모 유전자에서 자식 유전자 생성을 위한 함수
def birth_of_child(dad, mom, cut_part,length_of_chromosome):
    cut_dad = dad[0:cut_part]
    cut_mom = mom[cut_part:length_of_chromosome]
    return np.concatenate((cut_dad,cut_mom), axis=None)

# 부모 유전자 2개를 선택해 서로 crossover 해준다.(기존의 선택된 유전자 유지)
def crossover_chromosome(multi_chromosome,fitness_score):

    num_of_chromosome = len(multi_chromosome)
    crossover_chromosome = []
    # 상위 10개의 유전자를 선택해준다.
    good_chromosome = selection_pick(multi_chromosome,fitness_score)
    num_of_good = len(good_chromosome)

    # 세대 별 최적의 해 출력
    print(good_chromosome[0])

# Crossover - Pair ( 선택된 10개의 유전자들에 대해서 i, i+1번째로 Pair 해준다. )
    for i in range(0,num_of_good,2):     
        dad_chromosome = good_chromosome[i]
        mom_chromosome = good_chromosome[i+1]
# Crossover를 위해서 유전자를 자를 자리를 임의로 설정해준다.
# 선택된 2개의 부모 유전자와 cut을 매개변수로 하여 birth_of_child 함수를 통해 자식 유전자를 생성해준다. 
# 생성된 2개의 자식유전자를 crossover_chromosome 이라는 list에 넣는다.
        cut = random.randint(0,N_SIZE)
        first_child_chromosome = birth_of_child(dad_chromosome, mom_chromosome, cut,N_SIZE)
        second_child_chromosome = birth_of_child(mom_chromosome, dad_chromosome, cut,N_SIZE)
        crossover_chromosome.append(first_child_chromosome)
        crossover_chromosome.append(second_child_chromosome)
# 기존의 부모 유전자들 또한 남겨주기 위해 list에 넣어준다.
    for i in range(num_of_good):
        crossover_chromosome.append(good_chromosome[i])
# 기존의 부모 유전자 + 생성된 자식 유전자들을 반환해준다.
    return crossover_chromosome

# crossover한 유전자들에 대해서 mutation 해준다.
def mutation(crossover_chromosome):

    length_of_crossover = len(crossover_chromosome)
    # mutation된 유전자들을 담을 list -> mutation_chromosome
    mutation_chromosome = []

    # 매개변수를 통해 넘겨받은 모든 유전자에 대해 mutation을 진행한다.
    for i in range(length_of_crossover):
        target_mutation = crossover_chromosome[i]
    # 유전자 내부의 각각의 원소에 대해 mutation 할건지에 대하여 확률적으로 선택한다.
    # True : mutation 한다. / False : mutation 하지 않는다.
        for j in range(len(target_mutation)):
            choice = random.choice([True,False])
            if(choice):
                x = random.randrange(1,N_SIZE+1)
                # 만약 임의의 수가 리스트의 원소와 같다면 다시 임의의 수를 생성한다.
                while (x == target_mutation[j]):
                    x = random.randrange(1,N_SIZE+1)
                target_mutation[j] = x
            else:
                continue
    # mutation 완료된 유전자를 리스트에 담아준다.        
        mutation_chromosome.append(target_mutation)

    return mutation_chromosome

# update_generation => 세대를 교채해준다.
def update_generation(crossover_chromosome, total_chromosome_size):
    # 크로스오버된 유전자들을 복사해준다.
    new_generation = copy.deepcopy(crossover_chromosome)

    # 0세대 유전자 갯수만큼의 새로운 유전자들이 생성될 때까지 mutation 해준다.
    while (len(new_generation) <= total_chromosome_size):
        new_generation.extend(mutation(crossover_chromosome))
        
    # 0세대 유전자 갯수만큼만 새로운 유전자들을 넘겨준다.
    return new_generation[0:total_chromosome_size]


if __name__ == "__main__":

    # 0 세대 초기화
    generation = initialization_chromosome(100)
    gen = 0
    fitness_score = []
    num_of_queens = N_SIZE

    # 정답 찾기
    while True:
        # 100개의 유전자들에 대한 적합도 평가를 한다.
        for i in range(100):
            score = fitness_evaluation(generation[i])
            fitness_score.append(score)
        # 해당 세대에 해답이 있다면 반복문을 탈출한다.
        # 0세대는 임의로 설정해주는 것이기 떄문에 해답이 있더라도 포함시키지 않았습니다.
        if (num_of_queens in fitness_score) and gen !=0:
            break
        else:
            # 반복문을 탈출하지 못할경우 세대가 증가하므로 gen을 1 증가킨다.
            gen += 1
            print(gen, " Generation")
            # Selection 및 Crossover
            crossover_selection = crossover_chromosome(generation, fitness_score)
            # Mutation 및 Update_generation
            generation = update_generation(crossover_selection, 100)
            # 적합도 점수를 담아놓은 배열을 비워준다.
            fitness_score.clear()

    #해를 찾았을 경우 -> 세대에서 해당하는 답을 찾아 출력한다.
    for solution in generation:
        if fitness_evaluation(solution) == num_of_queens:
            print("\n\nOne of Solution( ",gen+1," Generation )")
            print(solution)
            sol_visual = solution
            break    


    
 # 시각화 하기   

    # 시각화 하기 위해 2차원 배열을 생성 해줍니다.
    # 2차원 배열을 좌표평면처럼 재배열 해줍니다.
    # 2차원 평면에 색을 입힘으로써 체스판에 퀸을 표현해 줄 수 있습니다.
    first_visual = [1]*(N_SIZE*N_SIZE)
    solution_visual = np.reshape(first_visual, (N_SIZE,N_SIZE))

    
    # 배열을 좌표평면이라 생각하고 색을 입히기 위해 0과 1을 각각 번갈아 입력해줍니다.
    # 0 - White / 1 - Black을 나타냅니다.
    for i in range(N_SIZE):
        for j in range(N_SIZE):
            if(i+j) % 2 == 1:
                solution_visual[i][j] = 0
            else:
                continue

    # cmap을 통해서 배경색을 Blues로 설정해줌으로써, 0-White / 1-Blues 로 표현할 수 있습니다.
    plt.imshow(solution_visual, cmap='Blues')

    # X축 / Y축에 불필요한 것들을 숨겨줍니다.
    plt.xticks([]) # x축 좌표 숨김
    plt.yticks([]) # y축 좌표 숨김

    # 제목을 입력합니다.
    plt.title("One of solutions in ChessBoard", fontsize = 13)

    # 퀸의 위치에 따라 Text 색깔을 달리하여 Text가 배경색에 묻히지 않도록 해줍니다.
    # plt.text를 통해 text 위치를 조정해줍니다.
    # 퀸의 갯수만큼 반복문을 사용함으로써 사용되는 퀸을 모두표현 할 수 있습니다.
    for col in range(num_of_queens):
        row = sol_visual[col]-1

        # col , row를 통해 퀸을 좌표평면에 표시해줍니다.
        # 표시 시 배경색과 text 색깔이 겹치지 않도록 설정해줍니다.
        if (col+row) %2 == 0:
            queen_color = "white"
        else:
            queen_color = "black"

        # 퀸 출력    
        plt.text(col, row, '♕', horizontalalignment='center', verticalalignment='center',
                 weight = 'bold', size = 30, 
                 color=queen_color)

    # 실제 그림을 출력합니다.
    plt.show()
