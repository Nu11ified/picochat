use picochat_eval::gsm8k::{extract_answer, GsmQuestion, format_gsm_prompt};

#[test]
fn test_extract_answer_basic() {
    let response = "First we add 3 + 4 = 7. Then multiply by 2.\n#### 14";
    assert_eq!(extract_answer(response), Some("14".to_string()));
}

#[test]
fn test_extract_answer_with_comma() {
    let response = "The total is #### 1,234";
    assert_eq!(extract_answer(response), Some("1234".to_string()));
}

#[test]
fn test_extract_answer_negative() {
    let response = "#### -5";
    assert_eq!(extract_answer(response), Some("-5".to_string()));
}

#[test]
fn test_extract_answer_none() {
    let response = "I don't know the answer";
    assert_eq!(extract_answer(response), None);
}

#[test]
fn test_extract_answer_decimal() {
    let response = "#### 3.14";
    assert_eq!(extract_answer(response), Some("3.14".to_string()));
}

#[test]
fn test_format_gsm_prompt() {
    let exemplars = vec![
        GsmQuestion {
            question: "What is 2+3?".to_string(),
            answer: "2+3=5\n#### 5".to_string(),
        },
    ];
    let test_q = GsmQuestion {
        question: "What is 4+5?".to_string(),
        answer: "#### 9".to_string(),
    };

    let prompt = format_gsm_prompt(&exemplars, &test_q);
    assert!(prompt.contains("What is 2+3?"));
    assert!(prompt.contains("#### 5"));
    assert!(prompt.contains("What is 4+5?"));
    assert!(prompt.ends_with("A: Let's think step by step.\n"));
}
