def sellers_dist(str1, str2, cost_sub=0.5, cost_ins=1, cost_del=1):
    prev_row = list(range(len(str2) + 1))
        
    min_distance = float('inf')
    
    for i in range(len(str1)):
        curr_row = [i + 1]
        for j in range(len(str2)):
            cost = 0 if str1[i] == str2[j] else cost_sub
            curr_row.append(min(curr_row[j] + cost_ins, prev_row[j + 1] + cost_del, prev_row[j] + cost))
        
        if curr_row[-1] < min_distance:
            min_distance = curr_row[-1]
        
        prev_row = curr_row
    
    return float(min_distance)


def find_nearest_neighbors(query_substring, string, distance_threshold=2, n=4, length_threshold=0):
    nearest_neighbors = []
    
       
    for i in range(len(string)):
        for j in range(i + 1, len(string) + 1):
            substring = string[i:j]
            if len(substring) < len(query_substring) - length_threshold or len(substring) > len(query_substring) + length_threshold:
                continue
            
            distance = sellers_dist(query_substring, substring)
            
            if (distance <= distance_threshold) or (len(nearest_neighbors) < n):
                nearest_neighbors.append((substring, distance))
    
    nearest_neighbors.sort(key=lambda x: x[1])
    
    if n is not None:
        return nearest_neighbors[:n]
    else:
        return nearest_neighbors




if __name__=="__main__":
       
    query_substring = "apple"
    string = "ape apples banana ackle ssna pple"
    length_threshold = 3
    distance_threshold = 3
    n=4
    nearest_neighbors_by_threshold = find_nearest_neighbors(query_substring, string,distance_threshold=distance_threshold, length_threshold=length_threshold)
    nearest_neighbors_by_number = find_nearest_neighbors(query_substring, string,n=n, length_threshold=length_threshold)

    print(f"The nearest neighbors for '{query_substring}' within an edit distance of the {distance_threshold} and a length threshold of {length_threshold} in the string:")
    for neighbor, distance in nearest_neighbors_by_threshold:
        print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})")

    print(f"\n{n} nearest neighbors for '{query_substring}' with a length threshold of {length_threshold} in the string:")
    for neighbor, distance in nearest_neighbors_by_number:
        print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})")