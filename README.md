## Urađeno (Aleksandar Ristic RN 37/2021)
- Postavljen ceo projekat od nule:
  - CMake konfiguracija
  - Git repozitorijum
  - Povezivanje biblioteka
  - Struktura `src/` i `shaders/`
- Sistem za učitavanje šejdera iz fajlova (`chess.vert`, `chess.frag`) sa uniformama iz C++ koda - promenjeno (shaderi su sad vecinom u main-u)
- Proceduralna geometrija šahovske table sa debljinom (mesh generisan u kodu sa normalama i UV koordinatama)
- Proceduralna šahovska tekstura u shaderu (checkerboard, broj polja, osvetljenje i posebna boja za stranice)
- Particle system (pre nego sto se razvila cela logika aplikacije, cestice se sada okidaju kada beli top pojede crnog):
  - Generisanje cestica sa random brzinom, uglom i životnim vekom
  - Gravitacija i sudaranje sa gornjom pločom table
  - Gašenje čestica nakon isteka života
- GPU deo particle sistema (VS + FS za cestice: veličina, fade-out, boja)
- Tastatura (promenjeno radi logike aplikacije):
  - Uključivanje/isključivanje table i čestica (1, 2, 3)
  - Start/stop emitovanja (P/O)
  - Reset (R)
  - Izlaz (ESC)
- Kamera: podešena perspektivna projekcija, pozicija, target, up vektor, FOV
-  **Post-processing pipeline**:
   - Implementiran **framebuffer** i render na ekran preko fullscreen quad-a
   - Dodata podrška za **više post-processing efekata** u lancu (pipeline)
   - Implementiran **tonemapping** shader za kontrolu osvetljenja i kontrasta
   - Implementiran **bloom efekat** sa threshold-om i blur-om za svetle delove scene
   - Dodata tastaturna kontrola za uključivanje/isključivanje bloom-a ("B")
- Implementirana reprodukcija zvuka koriscenjem Win32 Multimedia API-a
- Implementiran Ambient Occluison - dodatna teksura koja se koristi u shaderima
- Implementirano renderovanje vode ispod sahovsle table
- Implementiran skybox
- Implementiran Volumetric light scattering (Godray)
- Implementiran Font and GUI rendering, GUIs in 3D space (iznad belog i crnog topa)

## Urađeno (Jovan Brasanac RN 89/2022)
- Implementirani novi objekti i njihove funkcionalnosti:
  - Beli i crni top na početku na suprotnim krajevima table
  - Hijerarhija elemenata i rotirajući elementi na vrhu obe figure
  - Senke pokretnih elemenata
  - Kretanje topova pomoću tastature
- Dodata celokupna logika igre:
  - Simple GUI pred pocetak runde
  - Tajmer
  - Ispis pobednika i povratak na meni
  - Okidanje particle eksplozije kada beli top pojede crnog (završe na istom polju table)
- Tastatura:
  - Kretanje belog topa (strelice)
  - Kretanje crnog topa (WASD)
- Iskorišćena dodatna tekstura
  - Uveden je roughness teksturni uzorak (proceduralno generisan u CPU-u), koji u šejderu kontroliše specular (Blinn-Phong) i shininess
- Proširen set aktivnih šejdera (dodati `mesh.vert`, `mesh.frag`, `shadow_depth.vert`, `shadow_depth.frag`) - promenjeno, vecina u main-u
- Implementirana viseća lampa kao centralni izvor svetlosti

---

## Ispunjeno iz specifikacije
- **Generalna 3D grafika** - tabla, figure, lampa, cestice...
  ### Tehnike:
- **Proceduralne teksture** - šahovski uzorak u fragment shaderu  
- **Animated particle systems** - čestice sa kretanjem i interakcijom
- **Spotlight shadows and point light shadows** - usmereno centralno svetlo iznad table + senke
  ### Ostalo:
- **Interaktivnost** - kretanje šahovskih figura, dugmići, logika igre, tajmer, GUI...
- **Hijerarhije objekata** - rotirajući elementi vezani za figure
- **Korišćenje drugih tekstura pored albedo** - roughness teksturni uzorak za dodatni shine
- **Post-processing pipeline + efekti** - fleksibilan pipeline sa tonemap-om i bloom efektom (sa tastaturnom kontrolom za uključivanje/isključivanje)
  ### Opciono:
- Implementiran i zvuk
- ## Projekat pokriva sve zahteve specifikacije :D

---

## Šta još treba da se uradi

### Nemanja
- Eventualno jos jedna tehnika
- Eventualno jos pipeline efekata
- README  

---
