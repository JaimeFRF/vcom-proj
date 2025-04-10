from IO.json_handler import *
from tqdm import tqdm

results = read_results()

def compare_matrix_with_result(user, solution):
    TP = FP = FN = TN = 0
    
    for i in range(len(user)):
        for j in range(len(user[i])):
            u, s = user[i][j], solution[i][j]
            
            if s == 0 and u == 0:
                TN += 1
            elif s != 0 and u != 0:
                TP += 1
            elif s != 0 and u == 0:
                FN += 1
            elif s == 0 and u != 0:
                FP += 1

    return TP, FP, FN, TN

    

def test_implementation(imgs_data, check_number_pieces=True):
    total_tp = total_fp = total_fn = total_tn = 0

    for nbr, i in enumerate(imgs_data):
        print("\n" + "-" * 50)  
        print(f"Results for Image {nbr+1}: {i['name']}")
        print("-" * 50)

        sol = next((item for item in results if item["name"] == i["name"]), None)
        assert sol is not None

        TP, FP, FN, TN = compare_matrix_with_result(i["metadata"]["chessboard_matrix"], sol["pieces"])

        total_tp += TP
        total_fp += FP
        total_fn += FN
        total_tn += TN

        """ 
        with tqdm(total=100, position=0, leave=True, bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} correct') as pbar:
            pbar.update(int(accuracy * 100))

        total_pieces = TP + FN  # Ground truth positives
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 1.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

        if check_number_pieces:
            expected_black, actual_black = sol["black_pieces"], i["metadata"]["total_black"]
            expected_white, actual_white = sol["white_pieces"], i["metadata"]["total_white"]

            if expected_black != actual_black:
                print(f"Mismatch in black pieces. Expected {expected_black}, got {actual_black}.")
            if expected_white != actual_white:
                print(f"Mismatch in white pieces. Expected {expected_white}, got {actual_white}.") """

    total_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 1.0
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    print("\n" + "=" * 50)  
    print(f"Overall Accuracy: {total_accuracy * 100:.2f}%")
    print(f"Overall Precision: {total_precision * 100:.2f}%")
    print(f"Overall Recall: {total_recall * 100:.2f}%")
    print(f"Overall F1 Score: {total_f1 * 100:.2f}%")
    print("=" * 50)
    print("✅ Analysis Complete for All Images.")
    print("=" * 50)


        


                