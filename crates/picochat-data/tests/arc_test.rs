use picochat_data::arc::{ArcQuestion, load_arc_jsonl, format_arc_prompt};

#[test]
fn test_arc_question_parse() {
    let json = r#"{"question":"What causes seasons?","choices":["Tilt of Earth","Distance from Sun","Moon phases","Wind"],"answer_key":"A"}"#;
    let q: ArcQuestion = serde_json::from_str(json).unwrap();
    assert_eq!(q.question, "What causes seasons?");
    assert_eq!(q.choices.len(), 4);
    assert_eq!(q.answer_key, "A");
}

#[test]
fn test_arc_question_answer_index() {
    let q = ArcQuestion {
        question: "Test".to_string(),
        choices: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        answer_key: "C".to_string(),
    };
    assert_eq!(q.answer_index(), Some(2));
}

#[test]
fn test_format_arc_prompt() {
    let exemplars = vec![
        ArcQuestion {
            question: "What is H2O?".to_string(),
            choices: vec!["Fire".into(), "Water".into(), "Air".into(), "Earth".into()],
            answer_key: "B".to_string(),
        },
    ];
    let test_q = ArcQuestion {
        question: "What is the Sun?".to_string(),
        choices: vec!["Star".into(), "Planet".into(), "Moon".into(), "Comet".into()],
        answer_key: "A".to_string(),
    };
    let prompt = format_arc_prompt(&exemplars, &test_q);
    assert!(prompt.contains("What is H2O?"));
    assert!(prompt.contains("(B) Water"));
    assert!(prompt.contains("Answer: B"));
    assert!(prompt.contains("What is the Sun?"));
    assert!(prompt.contains("(A) Star"));
    assert!(prompt.ends_with("Answer: "));
}

#[test]
fn test_load_arc_from_string() {
    let data = r#"{"question":"Q1","choices":["a","b","c","d"],"answer_key":"A"}
{"question":"Q2","choices":["w","x","y","z"],"answer_key":"D"}"#;
    let questions: Vec<ArcQuestion> = data.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[1].answer_key, "D");
}
