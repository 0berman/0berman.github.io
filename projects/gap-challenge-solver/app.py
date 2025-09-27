import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from itertools import combinations
from streamlit_paste_button import paste_image_button as pbutton

# --- App Configuration ---
st.set_page_config(page_title="Gap Challenge Solver", layout="wide")

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CONFIDENCE_THRESHOLD_SHAPE = 0.60
SURENESS_THRESHOLD = 0.88
BLANK_STD_DEV_THRESHOLD = 15.0   
WHITE_THRESHOLD = 240
BASE_TEMPLATE_DIR = "templates"

def find_empty(board):
    """Finds the first empty cell ('blank') in the board."""
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 'blank':
                return (r, c)
    return None

def solve_with_backtracking(board, all_shapes):
    """Solves the puzzle using a recursive backtracking algorithm."""
    find = find_empty(board)
    if not find: return True
    row, col = find
    for shape in all_shapes:
        if shape in board[row] or shape in [board[i][col] for i in range(len(board))]: continue
        board[row][col] = shape
        if solve_with_backtracking(board, all_shapes): return True
        board[row][col] = 'blank'
    return False

def find_question_mark_solution(board, universe_of_shapes):
    """
    Finds the single shape that should replace the '6question' mark by trying
    all valid combinations of shapes for the given grid size.
    """
    question_pos = None
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == '6question':
                question_pos = (r, c); break
        if question_pos: break
    if not question_pos:
        st.error("No '6question' mark found on the board."); return None, None
    
    grid_size = len(board)
    if len(universe_of_shapes) < grid_size:
        st.warning(f"Warning: Only found {len(universe_of_shapes)} unique shapes for a {grid_size}x{grid_size} grid. Solution may be incorrect.")
        return None, None

    shape_combinations = combinations(universe_of_shapes, grid_size)
    solution_details = st.expander("Show Solver Attempts")

    for shape_combo in shape_combinations:
        shapes_to_try = list(shape_combo)
        solution_details.write(f"Attempting to solve with shape set: `{shapes_to_try}`")
        board_copy = [row[:] for row in board]
        qr, qc = question_pos
        board_copy[qr][qc] = 'blank'
        initial_board_valid = all(
            cell == 'blank' or cell in shapes_to_try 
            for row in board_copy for cell in row
        )
        if not initial_board_valid:
            solution_details.write("  - Skipping: Initial board contains shapes not in this set.")
            continue
        if solve_with_backtracking(board_copy, shapes_to_try):
            solution_details.success("  - SUCCESS: Solver found a complete solution with this set.")
            solution_shape = board_copy[qr][qc]
            return solution_shape, board_copy
        else:
            solution_details.write("  - FAILED: This combination does not lead to a solution.")
    st.error("Solver could not find a valid solution with any combination of shapes.")
    return None, None

# ==============================================================================
# --- IMAGE RECOGNITION LOGIC (from standalone_debugger.py) ---
# ==============================================================================

def crop_to_grid(source_image: np.ndarray):
    """Crops the image to the content's bounding box."""
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    non_zero_pixels = cv2.findNonZero(thresh)
    if non_zero_pixels is None: return source_image
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    return source_image[y:y+h, x:x+w]

def recognize_shape_in_cell(cell_roi_color, templates):
    """Helper function to find the best matching shape in a given cell ROI."""
    cell_roi_gray = cv2.cvtColor(cell_roi_color, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(cell_roi_gray)
    if std_dev < BLANK_STD_DEV_THRESHOLD: return "blank"
    
    best_match = {'label': 'blank', 'score': -1.0}
    early_exit = False

    for label, template_with_alpha in templates.items():
        if template_with_alpha.shape[2] == 4:
            mask = template_with_alpha[:,:,3]
            template_color = cv2.cvtColor(template_with_alpha, cv2.COLOR_BGRA2BGR)
        else:
            template_color = template_with_alpha
            template_gray_for_mask = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(template_gray_for_mask, 10, 255, cv2.THRESH_BINARY)
        
        h, w, _ = template_color.shape
        for scale in np.linspace(0.5, 2.0, 20):
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            if not (scaled_h > 0 and scaled_w > 0 and scaled_h <= cell_roi_color.shape[0] and scaled_w <= cell_roi_color.shape[1]): continue
            
            scaled_template = cv2.resize(template_color, (scaled_w, scaled_h))
            scaled_mask = cv2.resize(mask, (scaled_w, scaled_h))
            result = cv2.matchTemplate(cell_roi_color, scaled_template, cv2.TM_CCOEFF_NORMED, mask=scaled_mask)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if not np.isfinite(max_val): max_val = -1.0
            if max_val > best_match['score']:
                best_match.update({'score': max_val, 'label': label})
            
            # --- OPTIMIZATION: Early exit if we find a very confident match ---
            if max_val > SURENESS_THRESHOLD:
                early_exit = True
                break # Exit scale loop
        
        if early_exit:
            break # Exit shape loop
            
    final_label = 'blank'
    if best_match['score'] > CONFIDENCE_THRESHOLD_SHAPE:
        final_label = best_match['label']
    return final_label

def recognize_grid_and_options(image: Image.Image, grid_size: int, templates):
    """Recognizes shapes in the grid and options row, returning the board and shape universe."""
    original_image = np.array(image.convert('RGB'))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    border_size = 20
    bordered_image = cv2.copyMakeBorder(original_image, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cropped_color = crop_to_grid(bordered_image)
    img_h, img_w, _ = cropped_color.shape
    estimated_total_rows = grid_size + 1.1 
    estimated_cell_h = img_h / estimated_total_rows
    estimated_grid_h = int(estimated_cell_h * grid_size)
    puzzle_area = cropped_color[:estimated_grid_h, :]
    options_area = cropped_color[estimated_grid_h:, :]
    puzzle_area = crop_to_grid(puzzle_area)
    standard_size = 600
    aligned_color_image = cv2.resize(puzzle_area, (standard_size, standard_size))
    cell_height, cell_width = standard_size // grid_size, standard_size // grid_size
    output_grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            cell_roi = aligned_color_image[r*cell_height:(r+1)*cell_height, c*cell_width:(c+1)*cell_width]
            output_grid[r][c] = recognize_shape_in_cell(cell_roi, templates)
    
    universe_of_shapes = []
    if options_area.shape[0] > 10:
        num_options = grid_size
        cell_w_options = options_area.shape[1] // num_options
        for i in range(num_options):
            option_roi = options_area[:, i*cell_w_options:(i+1)*cell_w_options]
            option_roi = cv2.resize(option_roi, (cell_width, cell_height))
            shape_in_option = recognize_shape_in_cell(option_roi, templates)
            if shape_in_option not in ['blank', '6question']:
                universe_of_shapes.append(shape_in_option)
    
    for r in range(grid_size):
        for c in range(grid_size):
            shape_in_grid = output_grid[r][c]
            if shape_in_grid not in ['blank', '6question']:
                universe_of_shapes.append(shape_in_grid)
    
    if not universe_of_shapes:
        universe_of_shapes = ['1circle','2triangle','3square','4cross','5star']
    
    return output_grid, sorted(list(set(universe_of_shapes)))

# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================

st.title("🧩 Gap Challenge Solver")
st.info("To solve, take a screenshot, click the paste button below, and press Ctrl+V (or Cmd+V).")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    grid_size = st.radio("Grid Size", (4, 5), index=0)
    is_aon = st.toggle("AON Puzzle", value=True, help="Switch off for Practice puzzles.")
    
# --- Main App Body ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Puzzle")
    paste_result = pbutton("📋 Paste an image")
    if "pasted_image" not in st.session_state:
        st.session_state.pasted_image = None
    if paste_result.image_data is not None:
        st.session_state.pasted_image = paste_result.image_data
    if st.session_state.pasted_image:
        st.image(st.session_state.pasted_image, caption="Pasted from Clipboard", use_container_width=True)
    else:
        st.write("Awaiting a pasted image...")

with col2:
    st.subheader("Solution")
    if st.session_state.pasted_image:
        with st.spinner("Analyzing puzzle..."):
            image = st.session_state.pasted_image
            template_dir = BASE_TEMPLATE_DIR if is_aon else os.path.join(BASE_TEMPLATE_DIR, "practice")
            st.write(f"Using templates from: `{template_dir}`")
            shape_labels = ['1circle','2triangle','3square','4cross','5star','6question']
            try:
                templates = {
                    label: cv2.imread(os.path.join(template_dir, f"{label}.png"), cv2.IMREAD_UNCHANGED)
                    for label in shape_labels
                }
                if any(t is None for t in templates.values()):
                    st.error(f"One or more template files are missing from '{template_dir}'."); st.stop()
            except Exception as e:
                st.error(f"Error loading template files: {e}"); st.stop()
            
            initial_grid, detected_shapes = recognize_grid_and_options(image, grid_size, templates)
            
            if initial_grid and detected_shapes:
                st.write("Detected Initial Grid:")
                st.table(initial_grid)
                solution_shape, solved_grid = find_question_mark_solution(initial_grid, detected_shapes)
                if solution_shape and solved_grid:
                    st.success("Solution Found!")
                    st.metric(label="The shape for the '?' is", value=solution_shape)
                    st.write("Completed Grid:")
                    st.table(solved_grid)
                else:
                    st.error("Could not find a valid solution for this puzzle.")
    else:
        st.write("Click the paste button and press Ctrl+V to see the solution.")
