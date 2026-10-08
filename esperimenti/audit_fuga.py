"""La soglia 0.60 basta a impedire la fuga di supervisione nel vocabolario esteso?

`vocabolario_esteso.py` usa 600 parole del dizionario di sistema come condizioni ausiliarie a
pseudo-etichette. Il rischio: se il dizionario contiene una parola quasi sinonima di un
attributo TENUTO FUORI (`smiling` mentre `Smiling` e' held-out), addestrarsi su quella parola
e' contrabbando. Il regime \"stretto\" esclude le direzioni con coseno > 0.60 verso un
attributo tenuto fuori; qui si controlla se quella soglia e' sufficiente.

Per ognuno dei 3 split stampa, per ogni attributo tenuto fuori, le 3 parole del dizionario piu'
vicine con il loro coseno e quante superano 0.60. Serve a leggere il margine, non a decidere:
la conclusione registrata e' che il filtro cattura i casi netti (`cheerfully` 0.74 per Smiling)
ma ne lascia passare di piu' tenui (`enthusiasm` 0.56), quindi il regime stretto va letto come
*quasi* pulito, non pulito (`docs/piano-interpretabilita.md`, sez. 6d).

Diagnostica: stampa e basta, **nessun file di risultato salvato**.
Richiede `/usr/share/dict/american-english` e il checkpoint
`contributo-candidato/checkpoints/models_seed2026.pt`, che non passa da git.

Uso:  HF_HUB_OFFLINE=1 python esperimenti/audit_fuga.py
"""
import sys, random, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
ck = torch.load(ROOT/"contributo-candidato/checkpoints/models_seed2026.pt", map_location="cpu")
dtn = ck["dirs_text"]; dtn = dtn/dtn.norm(dim=-1, keepdim=True); nomi = list(ck["attr_names"])
parole = [w.strip().lower() for w in open("/usr/share/dict/american-english")
          if w.strip().isalpha() and 4 <= len(w.strip()) <= 12 and w.strip().islower()]
random.Random(0).shuffle(parole); parole = parole[:600]
from transformers import CLIPModel, CLIPTokenizer
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
tk = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
T = [("a photo of a person who is {}","a photo of a person who is not {}"),
     ("a photo of a person with {}","a photo of a person without {}"),
     ("a face with {}","a face without {}"),("{}","not {}")]
@torch.no_grad()
def enc(fr):
    o = m.get_text_features(**tk(fr, padding=True, return_tensors="pt"))
    f = (o.pooler_output if hasattr(o,"pooler_output") else o).float()
    return f/f.norm(dim=-1, keepdim=True)
W = []
for i in range(0, len(parole), 100):
    lot = parole[i:i+100]
    P = torch.stack([enc([t.format(w) for t,_ in T]).mean(0) for w in lot])
    N = torch.stack([enc([t.format(w) for _,t in T]).mean(0) for w in lot])
    v = P - N; W.append(v/v.norm(dim=-1, keepdim=True))
    print(f"  {min(i+100,len(parole))}/{len(parole)}", flush=True)
W = torch.cat(W)
print("\n" + "=" * 88)
print("LA SOGLIA 0.60 BASTA A IMPEDIRE LA FUGA DI SUPERVISIONE?")
print("=" * 88)
for split in (0, 1, 2):
    HOLD = sorted(random.Random(split).sample(
        [i for i in range(40) if nomi[i].strip()], 10))
    print(f"\n--- split {split}")
    for a in HOLD:
        s = W @ dtn[a]
        top = s.abs().topk(3).indices
        voci = ", ".join(f"{parole[i]} {float(s[i]):+.2f}" for i in top)
        n = int((s.abs() > 0.60).sum())
        flag = "  <-- ESCLUSE" if n else ""
        print(f"  {nomi[a]:<22}{voci:<52}[{n} sopra 0.60]{flag}")
