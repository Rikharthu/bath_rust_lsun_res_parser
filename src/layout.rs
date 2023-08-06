use once_cell::sync::Lazy;

pub const ROOM_TYPES: Lazy<Vec<RoomLayoutType>> = Lazy::new(|| {
    vec![
        RoomLayoutType {
            typeid: 0,
            cornermap: vec![1, 2, 3, 4, 5, 6, 7, 8],
            lines: vec![
                (1, 2),
                (3, 4),
                (5, 6),
                (7, 8),
                (1, 3),
                (3, 5),
                (5, 7),
                (7, 1),
            ],
        },
        RoomLayoutType {
            typeid: 1,
            cornermap: vec![3, 1, 4, 5, 7, 6],
            lines: vec![(1, 2), (1, 3), (1, 4), (4, 5), (4, 6)],
        },
        RoomLayoutType {
            typeid: 2,
            cornermap: vec![1, 2, 3, 7, 8, 5],
            lines: vec![(1, 2), (1, 3), (1, 4), (4, 5), (4, 6)],
        },
        RoomLayoutType {
            typeid: 3,
            cornermap: vec![7, 1, 5, 8],
            lines: vec![(1, 2), (1, 3), (1, 4)],
        },
        RoomLayoutType {
            typeid: 4,
            cornermap: vec![5, 3, 7, 6],
            lines: vec![(1, 2), (1, 3), (1, 4)],
        },
        RoomLayoutType {
            typeid: 5,
            cornermap: vec![7, 1, 8, 5, 3, 6],
            lines: vec![(1, 2), (1, 3), (1, 4), (4, 5), (4, 6)],
        },
        RoomLayoutType {
            typeid: 6,
            cornermap: vec![1, 7, 3, 5],
            lines: vec![(1, 2), (3, 4)],
        },
        RoomLayoutType {
            typeid: 7,
            cornermap: vec![],
            lines: vec![(1, 2), (3, 4)],
        },
        RoomLayoutType {
            typeid: 8,
            cornermap: vec![1, 7],
            lines: vec![(1, 2)],
        },
        RoomLayoutType {
            typeid: 9,
            cornermap: vec![3, 5],
            lines: vec![(1, 2)],
        },
        RoomLayoutType {
            typeid: 10,
            cornermap: vec![7, 5],
            lines: vec![(1, 2)],
        },
    ]
});

#[derive(Debug, Clone)]
pub struct RoomLayoutType {
    pub typeid: u8,
    pub cornermap: Vec<u8>,
    pub lines: Vec<(u8, u8)>,
    // pub region // Doesn't seem to be used
}
