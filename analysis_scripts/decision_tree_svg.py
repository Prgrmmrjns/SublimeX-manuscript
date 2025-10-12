def create_simple_tree(filename):
    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg.append('<svg width="550" height="440" xmlns="http://www.w3.org/2000/svg">')
    
    node_color = '#f9f1f9'
    edge_color = '#000000'
    text_color = '#000000'
    yes_color = '#4CAF50'
    no_color = '#f44336'
    leaf_color_0 = '#FFE5E5'
    leaf_color_1 = '#E5FFE5'
    
    level_height = 120
    node_width = 130
    node_height = 55
    
    # Root node - Pattern 1
    root_x, root_y = 250, 20
    svg.append(f'<rect x="{root_x - node_width/2}" y="{root_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{root_x}" y="{root_y + 23}" text-anchor="middle" font-size="13" font-weight="bold" fill="{text_color}">Pattern 1</text>')
    svg.append(f'<text x="{root_x}" y="{root_y + 42}" text-anchor="middle" font-size="11" fill="{text_color}">RMSE ≤ 0.52?</text>')
    
    # Left branch - Class 0 leaf
    leaf1_x, leaf1_y = 130, root_y + level_height
    svg.append(f'<line x1="{root_x - 30}" y1="{root_y + node_height}" x2="{leaf1_x}" y2="{leaf1_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="175" y="{root_y + level_height - 20}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{leaf1_x - node_width/2}" y="{leaf1_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf1_x}" y="{leaf1_y + 28}" text-anchor="middle" font-size="13" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{leaf1_x}" y="{leaf1_y + 45}" text-anchor="middle" font-size="10" fill="{text_color}">n = 82</text>')
    
    # Right branch - Pattern 2 node
    pattern2_x, pattern2_y = 370, root_y + level_height
    svg.append(f'<line x1="{root_x + 30}" y1="{root_y + node_height}" x2="{pattern2_x}" y2="{pattern2_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="325" y="{root_y + level_height - 20}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{pattern2_x - node_width/2}" y="{pattern2_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{pattern2_x}" y="{pattern2_y + 23}" text-anchor="middle" font-size="13" font-weight="bold" fill="{text_color}">Pattern 2</text>')
    svg.append(f'<text x="{pattern2_x}" y="{pattern2_y + 42}" text-anchor="middle" font-size="11" fill="{text_color}">RMSE ≤ 0.38?</text>')
    
    # Pattern 2 left branch - Class 1 leaf
    leaf2_x, leaf2_y = 300, pattern2_y + level_height
    svg.append(f'<line x1="{pattern2_x - 30}" y1="{pattern2_y + node_height}" x2="{leaf2_x}" y2="{leaf2_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="325" y="{pattern2_y + level_height - 20}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{leaf2_x - node_width/2}" y="{leaf2_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_1}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf2_x}" y="{leaf2_y + 28}" text-anchor="middle" font-size="13" font-weight="bold" fill="{text_color}">Class 1</text>')
    svg.append(f'<text x="{leaf2_x}" y="{leaf2_y + 45}" text-anchor="middle" font-size="10" fill="{text_color}">n = 54</text>')
    
    # Pattern 2 right branch - Class 0 leaf
    leaf3_x, leaf3_y = 440, pattern2_y + level_height
    svg.append(f'<line x1="{pattern2_x + 30}" y1="{pattern2_y + node_height}" x2="{leaf3_x}" y2="{leaf3_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="415" y="{pattern2_y + level_height - 20}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{leaf3_x - node_width/2}" y="{leaf3_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf3_x}" y="{leaf3_y + 28}" text-anchor="middle" font-size="13" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{leaf3_x}" y="{leaf3_y + 45}" text-anchor="middle" font-size="10" fill="{text_color}">n = 34</text>')
    
    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))
    
    print(f"Simple tree saved to {filename}")

def create_complex_tree(filename):
    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg.append('<svg width="1200" height="650" xmlns="http://www.w3.org/2000/svg">')
    
    node_color = '#f9f1f9'
    edge_color = '#000000'
    text_color = '#000000'
    yes_color = '#4CAF50'
    no_color = '#f44336'
    leaf_color_0 = '#FFE5E5'
    leaf_color_1 = '#E5FFE5'
    
    level_height = 130
    node_width = 120
    node_height = 50
    
    root_x, root_y = 600, 20
    svg.append(f'<rect x="{root_x - node_width/2}" y="{root_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{root_x}" y="{root_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 1</text>')
    svg.append(f'<text x="{root_x}" y="{root_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.65?</text>')
    
    left1_x, left1_y = 350, root_y + level_height
    svg.append(f'<line x1="{root_x - 25}" y1="{root_y + node_height}" x2="{left1_x}" y2="{left1_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="450" y="{root_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{left1_x - node_width/2}" y="{left1_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{left1_x}" y="{left1_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 2</text>')
    svg.append(f'<text x="{left1_x}" y="{left1_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.42?</text>')
    
    right1_x, right1_y = 850, root_y + level_height
    svg.append(f'<line x1="{root_x + 25}" y1="{root_y + node_height}" x2="{right1_x}" y2="{right1_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="750" y="{root_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{right1_x - node_width/2}" y="{right1_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{right1_x}" y="{right1_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 3</text>')
    svg.append(f'<text x="{right1_x}" y="{right1_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.78?</text>')
    
    left2_x, left2_y = 200, left1_y + level_height
    svg.append(f'<line x1="{left1_x - 35}" y1="{left1_y + node_height}" x2="{left2_x}" y2="{left2_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="260" y="{left1_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{left2_x - node_width/2}" y="{left2_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{left2_x}" y="{left2_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 4</text>')
    svg.append(f'<text x="{left2_x}" y="{left2_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.31?</text>')
    
    right2_x, right2_y = 500, left1_y + level_height
    svg.append(f'<line x1="{left1_x + 35}" y1="{left1_y + node_height}" x2="{right2_x}" y2="{right2_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="440" y="{left1_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{right2_x - node_width/2}" y="{right2_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{right2_x}" y="{right2_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 5</text>')
    svg.append(f'<text x="{right2_x}" y="{right2_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.55?</text>')
    
    left3_x, left3_y = 700, right1_y + level_height
    svg.append(f'<line x1="{right1_x - 35}" y1="{right1_y + node_height}" x2="{left3_x}" y2="{left3_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="760" y="{right1_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{left3_x - node_width/2}" y="{left3_y}" width="{node_width}" height="{node_height}" fill="{node_color}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="{left3_x}" y="{left3_y + 22}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Pattern 6</text>')
    svg.append(f'<text x="{left3_x}" y="{left3_y + 38}" text-anchor="middle" font-size="10" fill="{text_color}">RMSE ≤ 0.48?</text>')
    
    right3_x, right3_y = 1000, right1_y + level_height
    svg.append(f'<line x1="{right1_x + 35}" y1="{right1_y + node_height}" x2="{right3_x}" y2="{right3_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="940" y="{right1_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{right3_x - node_width/2}" y="{right3_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{right3_x}" y="{right3_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{right3_x}" y="{right3_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 28</text>')
    
    leaf1_x, leaf1_y = 100, left2_y + level_height
    svg.append(f'<line x1="{left2_x - 30}" y1="{left2_y + node_height}" x2="{leaf1_x}" y2="{leaf1_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="135" y="{left2_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{leaf1_x - node_width/2}" y="{leaf1_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf1_x}" y="{leaf1_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{leaf1_x}" y="{leaf1_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 35</text>')
    
    leaf2_x, leaf2_y = 300, left2_y + level_height
    svg.append(f'<line x1="{left2_x + 30}" y1="{left2_y + node_height}" x2="{leaf2_x}" y2="{leaf2_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="265" y="{left2_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{leaf2_x - node_width/2}" y="{leaf2_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_1}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf2_x}" y="{leaf2_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 1</text>')
    svg.append(f'<text x="{leaf2_x}" y="{leaf2_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 32</text>')
    
    leaf3_x, leaf3_y = 420, right2_y + level_height
    svg.append(f'<line x1="{right2_x - 30}" y1="{right2_y + node_height}" x2="{leaf3_x}" y2="{leaf3_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="450" y="{right2_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{leaf3_x - node_width/2}" y="{leaf3_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_1}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf3_x}" y="{leaf3_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 1</text>')
    svg.append(f'<text x="{leaf3_x}" y="{leaf3_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 41</text>')
    
    leaf4_x, leaf4_y = 580, right2_y + level_height
    svg.append(f'<line x1="{right2_x + 30}" y1="{right2_y + node_height}" x2="{leaf4_x}" y2="{leaf4_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="550" y="{right2_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{leaf4_x - node_width/2}" y="{leaf4_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf4_x}" y="{leaf4_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{leaf4_x}" y="{leaf4_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 38</text>')
    
    leaf5_x, leaf5_y = 700, left3_y + level_height
    svg.append(f'<line x1="{left3_x - 30}" y1="{left3_y + node_height}" x2="{leaf5_x}" y2="{leaf5_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="735" y="{left3_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{yes_color}">Yes</text>')
    
    svg.append(f'<rect x="{leaf5_x - node_width/2}" y="{leaf5_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_1}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf5_x}" y="{leaf5_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 1</text>')
    svg.append(f'<text x="{leaf5_x}" y="{leaf5_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 45</text>')
    
    leaf6_x, leaf6_y = 900, left3_y + level_height
    svg.append(f'<line x1="{left3_x + 30}" y1="{left3_y + node_height}" x2="{leaf6_x}" y2="{leaf6_y}" stroke="{edge_color}" stroke-width="2"/>')
    svg.append(f'<text x="865" y="{left3_y + level_height - 25}" text-anchor="middle" font-size="11" font-weight="bold" fill="{no_color}">No</text>')
    
    svg.append(f'<rect x="{leaf6_x - node_width/2}" y="{leaf6_y}" width="{node_width}" height="{node_height}" fill="{leaf_color_0}" stroke="{edge_color}" stroke-width="2" rx="5" ry="5"/>')
    svg.append(f'<text x="{leaf6_x}" y="{leaf6_y + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="{text_color}">Class 0</text>')
    svg.append(f'<text x="{leaf6_x}" y="{leaf6_y + 40}" text-anchor="middle" font-size="10" fill="{text_color}">n = 51</text>')
    
    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))
    
    print(f"Complex tree saved to {filename}")

if __name__ == '__main__':
    create_simple_tree('../manuscript/images/flowchart/decision_tree_simple.svg')
    create_complex_tree('../manuscript/images/flowchart/decision_tree_complex.svg')
