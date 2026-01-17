import os
import json
import argparse
from tqdm import tqdm
import re

# ------------------------------------------------------------------------------
# Configuration & Prompt Templates
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert educational discourse analyst. Your task is to segment a tutoring dialogue into coherent topics.

Definition:
- Boundary index = the 0-indexed turn number of the last utterance in a segment.
- Segments must be contiguous and cover the entire dialogue.
- Always include the final turn index as the last boundary.

Add a boundary when:
- The dialogue moves to a new problem/question, or a new subtask with a different goal.
- The interaction shifts between social talk and lesson content.

Do NOT add a boundary when:
- The tutor restates the same idea, checks understanding, or the student gives brief answers within the same topic.

Use the following construct definitions to decide where meaningful boundaries occur.
Do NOT label the dialogue with these constructs; use them only to guide segmentation.

### Subcategory 1: Feedback Loops
Focus on the quality of feedback in the tutoring session, which is defined as the degree to which feedback advances learning and understanding and encourages student participation. A feedback loop refers to a sustained back-and-forth exchange between a teacher and a student that builds on prior turns and pushes learning forward.
- Low Quality: Feedback is mostly absent, minimal, or one-sided. Exchanges are brief, superficial, and do not build on student thinking.
- Mid Quality: Feedback loops occur occasionally, but are inconsistent or limited in depth. Some back-and-forth exchanges involve clarifying questions or brief elaboration.
- High Quality: Feedback loops are frequent and often extend or deepen student understanding. The teacher regularly engages in meaningful back-and-forth exchanges that build on student thinking.

### Subcategory 2: Scaffolding
Scaffolding refers to the teacher’s use of hints, prompts, or structured support that helps students move toward successful task completion or deeper understanding. It is not however an explanation of a concept. Scaffolding enables students to perform at a higher level than they could independently by breaking down tasks, modeling thinking, or guiding problem-solving steps.
- Low Quality: Students are not provided with meaningful assistance, hints, or prompting, and are often left to complete work on their own.
- Mid Quality: The teacher sometimes scaffolds student learning, but these interactions are often brief or shallow.
- High Quality: The teacher frequently scaffolds student learning, enabling students to perform at a higher level than they could independently. Support is purposeful and leads to visible progress.

### Subcategory 3: Building on Student Responses
Building on student responses refers to the teacher’s practice of expanding, instances of clarification and specific feedback, or refining what a student says to further their understanding. This includes asking targeted follow-up questions, elaborating on partial answers, or redirecting thinking in a way that pushes learning forward. It is not praise, repetition, or scaffolding.
- Low Quality: The teacher gives no or very general feedback. They accept student answers and move on without clarifying or extending.
- Mid Quality: The teacher occasionally expands or clarifies student responses, but these exchanges are brief or shallow.
- High Quality: The teacher frequently builds on student responses with meaningful expansions. This includes giving specific feedback, asking clarifying or elaborative questions, and encouraging deeper reasoning.

### Subcategory 4: Encouragement and Affirmation
Encouragement and affirmation refer to the teacher’s recognition of student effort and encouragement of persistence, often through specific praise or motivational feedback that supports continued engagement in the learning process. Generic statements such as “Great,” “Okay,” “That’s correct,” or “Glad I could help” do not count as encouragement or affirmation.
- Low Quality: The teacher gives little or no encouragement. Feedback is correctness-focused and does not recognize student effort or struggle.
- Mid Quality: The teacher occasionally encourages students with brief, general comments (e.g., “Nice try,” “Keep going”).
- High Quality: Encouragement targets student effort, strategies, or perseverance. It helps students persist during challenges and feel supported.


Output format (strict)
Return ONLY a JSON object in the following form (no extra text, no markdown):
{"boundary_indices":[integer, integer, ...]}

"""

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def load_dialogue(file_path):
    """
    Reads a dialogue file and returns a list of utterances.
    Matches the logic of the traditional segmentation script.
    """
    text = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "=======" in line:
                    continue
                text.append(line)
    except Exception as e:
        print(f"[WARN] Failed to read file {file_path}: {e}")
        return None
    return text

def call_llm(prompt, dialogue_text, model_name="gpt-5"):
    """
    Calls the LLM to get segmentation results.
    Supports Cornell AI Gateway or standard OpenAI.
    """
    import os
    from openai import OpenAI

    # 1. Try AI Gateway Config
    gateway_url = os.getenv("AI_GATEWAY_BASE_URL")
    gateway_key = os.getenv("AI_GATEWAY_KEY")
    gateway_provider = os.getenv("AI_GATEWAY_PROVIDER")
    
    # 2. Try Standard OpenAI Config
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")

    client = None
    model_to_use = model_name

    if gateway_url and gateway_key:
        # print(f"[INFO] Using AI Gateway at {gateway_url}")
        client = OpenAI(base_url=gateway_url, api_key=gateway_key)
        if gateway_provider:
            model_to_use = gateway_provider
    elif openai_key:
        # print(f"[INFO] Using standard OpenAI API")
        client = OpenAI(api_key=openai_key)
    
    if client:
        try:
            full_prompt = f"{prompt}\n\nHere is the dialogue to segment:\n\n{dialogue_text}\n\nRespond ONLY with the JSON object."
            
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                # Note: 'json_object' format might not be supported by all gateway models, 
                # but usually is for GPT-4 class models.
                # If it fails, we might need to remove it or try-catch.
                # response_format={"type": "json_object"} 
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return None
    
    # ------------------------------------------------------------------
    # MOCK RESPONSE (Fallback)
    # ------------------------------------------------------------------
    print(f"[INFO] No valid API configuration found. Mocking response...")
    
    # Create a dummy segmentation based on length
    mid_point = len(dialogue_text.split('\n')) // 2
    end_point = len(dialogue_text.split('\n')) - 1
    
    return json.dumps({
        "boundary_indices": [mid_point, end_point]
    })

def parse_llm_response(response_str, num_utterances):
    """
    Parses the LLM's string response into a structured dictionary.
    Calculates segment lengths from boundary indices.
    """
    try:
        # Attempt to find JSON block if wrapped in markdown code fences
        json_match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
        if json_match:
            response_str = json_match.group(1)
        
        data = json.loads(response_str)
        boundary_indices = data.get("boundary_indices", [])
        
        # Validate indices
        # They should be integers, sorted, and within range [0, num_utterances-1]
        boundary_indices = sorted([int(i) for i in boundary_indices if isinstance(i, (int, float))])
        boundary_indices = [i for i in boundary_indices if 0 <= i < num_utterances]
        
        # Calculate segment lengths
        segment_lengths = []
        last_idx = -1
        for idx in boundary_indices:
            length = idx - last_idx
            segment_lengths.append(length)
            last_idx = idx
            
        # If the last boundary is NOT the end of the dialogue, we might have a trailing segment?
        # Traditional script usually ensures the last utterance is a boundary or implicitly ends there.
        # If the LLM didn't include the last utterance index, we should probably add the remainder as a segment
        # OR assume the LLM only marks *internal* boundaries.
        # However, the prompt asks for the "last utterance in a segment".
        # If the dialogue ends at 5, and the last segment is 2-5, the boundary is 5.
        # Let's ensure the last utterance is covered.
        
        if boundary_indices and boundary_indices[-1] != num_utterances - 1:
            # Add the final segment
            length = (num_utterances - 1) - last_idx
            segment_lengths.append(length)
            # Optionally add the final index to boundary_indices if we want it to be exhaustive
            # The traditional script output shows boundary_indices ending with the last index?
            # Let's check the screenshot.
            # Screenshot: Dialogue has 22 items (0-21).
            # Boundary indices: 1, 2, 13, 18.
            # Segment lengths: 2, 1, 11, 5.
            # Sum of lengths = 2+1+11+5 = 19.
            # Wait, 22 items. 19 != 22.
            # Ah, the screenshot shows boundary indices 1, 2, 13, 18.
            # 0-1 (len 2)
            # 2-2 (len 1)
            # 3-13 (len 11)
            # 14-18 (len 5)
            # 19-21 (len 3) -> Missing from boundary_indices?
            # The screenshot shows "segment_lengths" has 5 items.
            # But "boundary_indices" has 4 items.
            # So the last segment (19-21) is implied and NOT in boundary_indices.
            pass
        elif not boundary_indices:
             # No boundaries found, whole dialogue is one segment
             segment_lengths = [num_utterances]
        
        # Re-calculate lengths carefully to match the screenshot logic
        # Screenshot logic seems to be:
        # segment 0: ends at boundary_indices[0]
        # segment 1: ends at boundary_indices[1]
        # ...
        # segment N: ends at end of dialogue (implicit)
        
        final_lengths = []
        prev = -1
        for b in boundary_indices:
            final_lengths.append(b - prev)
            prev = b
            
        # Check if there's a remainder
        if prev < num_utterances - 1:
            final_lengths.append((num_utterances - 1) - prev)
            
        return {
            "boundary_indices": boundary_indices,
            "segment_lengths": final_lengths
        }

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return {"error": "json_parse_error", "raw_response": response_str}
    except Exception as e:
        print(f"[ERROR] Error processing segments: {e}")
        return {"error": f"processing_error: {e}", "raw_response": response_str}

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

def process_folder(data_dir, out_json, model_name="gpt-5", limit=None):
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    input_files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and f != ".DS_Store"
    ]
    
    if limit:
        input_files = input_files[:limit]
        print(f"[INFO] Limiting execution to first {limit} files.")
    
    all_results = {}
    
    for file in tqdm(input_files, desc="LLM Segmentation"):
        path = os.path.join(data_dir, file)
        utterances = load_dialogue(path)
        
        if not utterances:
            continue
            
        # Format dialogue for prompt (numbered lines can help LLM)
        dialogue_text = "\n".join([f"{i}: {u}" for i, u in enumerate(utterances)])
        
        # Call LLM
        response_str = call_llm(SYSTEM_PROMPT, dialogue_text, model_name)
        
        if response_str:
            # Parse Results
            parsed = parse_llm_response(response_str, len(utterances))
            
            all_results[file] = {
                "utterances": utterances,
                **parsed
            }
        
    # Save to JSON
    os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved LLM predictions to {out_json}")

def main():
    parser = argparse.ArgumentParser(description="Run LLM-based dialogue segmentation.")
    parser.add_argument("--data_dir", required=True, help="Directory containing dialogue text files.")
    parser.add_argument("--out_json", required=True, help="Path to save output JSON.")
    parser.add_argument("--model", default="gpt-5", help="LLM model name to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process.")
    
    args = parser.parse_args()
    
    process_folder(args.data_dir, args.out_json, args.model, args.limit)

if __name__ == "__main__":
    main()
