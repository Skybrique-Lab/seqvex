use seqvex::foundation::observation::{Observation, SequenceNumber};

#[test]
fn observation_can_be_represented() {
    let observation = Observation::new(42_i64);
    assert_eq!(*observation.value(), 42);
}

#[test]
fn observation_preserves_required_input_information() {
    let observation = Observation::new(String::from("payload"));
    assert_eq!(observation.value(), "payload");
}

#[test]
fn observation_can_carry_sequence_context_when_required() {
    let without_context = Observation::new(1_i64);
    assert_eq!(without_context.sequence(), None);

    let with_context = Observation::new(1_i64).with_sequence(SequenceNumber::new(7));
    assert_eq!(with_context.sequence(), Some(SequenceNumber::new(7)));
}
