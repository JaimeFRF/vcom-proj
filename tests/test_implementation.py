from IO.json_handler import *
from tqdm import tqdm

results = read_results()

def compare_matrix_with_result(user, solution):
    wrongs = 0 
    rights = 0
    #piece_mapping = {0: "empty", 1: "white piece", 2: "black piece"}  
    
    for i in range(len(user)):
        for j in range(len(user[i])):
            if (user[i][j] != solution[i][j]):
                #print(f"Mismatch at ({i}, {j}): Expected {piece_mapping.get(solution[i][j], 'unknown')}, but got {piece_mapping.get(user[i][j], 'unknown')}")
                wrongs += 1
            elif (user[i][j] != 0): #Got it right and it is not an empty tile
                rights += 1
    return (wrongs, rights)
    

def test_implementation(imgs_data, check_number_pieces=True):
    total_acc = 0
    for nbr, i in enumerate(imgs_data):
        print("\n" + "-" * 50)  
        print(f"Results for Image {nbr+1}: {i['name']}")
        print("-" * 50)  # Another separator for clarity

        sol = next((item for item in results if item["name"] == i["name"]), None)
        assert sol is not None

        total_black, total_white = i["metadata"]["total_black"], i["metadata"]["total_white"]
        total_pieces = total_black + total_white

        mistakes, rights = compare_matrix_with_result(i["metadata"]["chessboard_matrix"], sol["pieces"])
        accuracy = max(0, (total_pieces - mistakes) / total_pieces) if total_pieces > 0 else 1.0
        total_acc += accuracy

        with tqdm(total=100, position=0, leave=True, bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} correct') as pbar:
            pbar.update(int(accuracy * 100))

        print(f"You missed {mistakes} pieces.")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"You got {rights} pieces right")

        if check_number_pieces:
            expected_black, actual_black = sol["black_pieces"], total_black
            expected_white, actual_white = sol["white_pieces"], total_white

            if expected_black != actual_black:
                print(f"Mismatch in black pieces. Expected {expected_black}, got {actual_black}.")
            if expected_white != actual_white:
                print(f"Mismatch in white pieces. Expected {expected_white}, got {actual_white}.")
    
    print("\n" + "=" * 50)  
    print(f"The overall accuracy is: {(total_acc / len(imgs_data)) * 100:.2f}%")
    print("=" * 50)
    print("✅ Analysis Complete for All Images.")
    print("=" * 50)


        


                