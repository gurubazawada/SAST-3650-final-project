from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    key: str
    pole_a: str
    pole_b: str
    label_a: str
    label_b: str
    description: str


PRESET_AXES: dict[str, Axis] = {
    a.key: a
    for a in [
        Axis(
            "action-renunciation",
            pole_a="action, karma-yoga, duty performed in the world",
            pole_b="renunciation, sannyasa, withdrawal from action",
            label_a="action",
            label_b="renunciation",
            description="karma-yoga vs. sannyasa — the book's central practical tension",
        ),
        Axis(
            "attachment-detachment",
            pole_a="attachment, desire, clinging to outcomes",
            pole_b="detachment, vairagya, freedom from craving",
            label_a="attachment",
            label_b="detachment",
            description="raga vs. vairagya",
        ),
        Axis(
            "knowledge-devotion",
            pole_a="knowledge, jnana, wisdom through understanding",
            pole_b="devotion, bhakti, love and surrender to the divine",
            label_a="knowledge",
            label_b="devotion",
            description="jnana-yoga vs. bhakti-yoga",
        ),
        Axis(
            "self-world",
            pole_a="the eternal Self, Atman, inner reality",
            pole_b="the changing world, samsara, outer phenomena",
            label_a="Self",
            label_b="world",
            description="Atman/Brahman vs. prakriti",
        ),
        Axis(
            "pleasure-pain",
            pole_a="pleasure, sukha, comfort, delight",
            pole_b="pain, duhkha, sorrow, suffering",
            label_a="pleasure",
            label_b="pain",
            description="the dualities of experience",
        ),
        Axis(
            "fear-fearlessness",
            pole_a="fear, anxiety, trembling before death",
            pole_b="fearlessness, abhaya, courage in crisis",
            label_a="fear",
            label_b="fearlessness",
            description="Arjuna's core transformation",
        ),
        Axis(
            "war-peace",
            pole_a="war, battle, striking down the enemy",
            pole_b="peace, serenity, stillness of mind",
            label_a="war",
            label_b="peace",
            description="the narrative frame",
        ),
        Axis(
            "desire-contentment",
            pole_a="desire, kama, longing for gratification",
            pole_b="contentment, santosha, fulfillment within",
            label_a="desire",
            label_b="contentment",
            description="kama vs. santosha",
        ),
        Axis(
            "ignorance-wisdom",
            pole_a="ignorance, avidya, delusion, confusion",
            pole_b="wisdom, vidya, insight, clear seeing",
            label_a="ignorance",
            label_b="wisdom",
            description="avidya vs. vidya",
        ),
        Axis(
            "body-spirit",
            pole_a="the mortal body, flesh, worn-out garment",
            pole_b="the immortal Self, that which dwells within",
            label_a="body",
            label_b="spirit",
            description="dehin vs. deha",
        ),
        Axis(
            "ego-surrender",
            pole_a="ego, ahamkara, pride, the 'I' and 'mine'",
            pole_b="surrender, sharanagati, taking refuge in the Lord",
            label_a="ego",
            label_b="surrender",
            description="ego vs. bhakti",
        ),
        Axis(
            "doubt-faith",
            pole_a="doubt, confusion, uncertainty of will",
            pole_b="faith, shraddha, steady conviction",
            label_a="doubt",
            label_b="faith",
            description="samshaya vs. shraddha",
        ),
        Axis(
            "sorrow-equanimity",
            pole_a="sorrow, grief, despair, lamentation",
            pole_b="equanimity, samatva, evenness in pleasure and pain",
            label_a="sorrow",
            label_b="equanimity",
            description="shoka vs. samatva",
        ),
        Axis(
            "multiplicity-unity",
            pole_a="the many, plural selves, difference, separation",
            pole_b="the one, non-duality, seeing the Self in all",
            label_a="multiplicity",
            label_b="unity",
            description="dvaita vs. advaita",
        ),
        Axis(
            "senses-selfcontrol",
            pole_a="the senses running wild, sense objects, appetite",
            pole_b="self-control, mastery of senses, the sage who restrains",
            label_a="senses",
            label_b="self-control",
            description="indriya-nigraha",
        ),
        Axis(
            "activity-stillness",
            pole_a="action, movement, effort, rajas",
            pole_b="stillness, meditation, silence, sattva",
            label_a="activity",
            label_b="stillness",
            description="gunas",
        ),
        Axis(
            "worldly-divine",
            pole_a="worldly reward, wealth, status, pleasure here and now",
            pole_b="the divine goal, liberation, union with the eternal",
            label_a="worldly",
            label_b="divine",
            description="pravritti vs. nivritti",
        ),
    ]
}
