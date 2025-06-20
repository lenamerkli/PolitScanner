from torch.utils.data import Dataset
from torch import tensor
from sentence_splitter import INPUT_SIZE
import typing as t


TRAINING_DATA = [
    {
        'text': b'Der Bundesrat wird beauftragt, den linksrheinischen NEAT-Zubringer Antwerpen\xe2\x80\x93Basel als zweite n\xc3\xb6rdliche Zulaufstrecke zeitnah auszubauen. Insbesondere soll die Schweiz die daf\xc3\xbcr erforderliche Profilanpassung der Vogesen-Tunnel vollst\xc3\xa4ndig finanzieren und hierf\xc3\xbcr vorrangig die freiwerdenden Mittel der Rollenden Landstrasse (RoLa) aus den Jahren 2026\xe2\x80\x932028 einsetzen.\nDie NEAT hat die Kapazit\xc3\xa4t des alpenquerenden G\xc3\xbctertransitverkehrs durch die Schweiz deutlich erh\xc3\xb6ht. Allerdings ist der wichtigste n\xc3\xb6rdliche Zulauf, die rechtsrheinische Rheintalbahn in Deutschland, bereits heute \xc3\xbcberlastet und \xc3\xa4usserst st\xc3\xb6ranf\xc3\xa4llig. Verz\xc3\xb6gerungen beim dortigen Ausbau (voraussichtliche Fertigstellung erst 2042) versch\xc3\xa4rfen dieses Engpassproblem. Um die Resilienz des Systems zu st\xc3\xa4rken, braucht es dringend einen zweiten Nordzulauf zur NEAT.',
        'data': [(140, 1), (374, 1), (479, 1), (634, 1), (750, 1), (847, 0)],
    },
    {
        'text': b'Der linksrheinische Korridor Antwerpen\xe2\x80\x93Basel bietet sich als Alternative an. Er verl\xc3\xa4uft durch die franz\xc3\xb6sische Region Grand Est und k\xc3\xb6nnte als zus\xc3\xa4tzlicher NEAT-Zubringer dienen. Allerdings stellen die bestehenden Vogesen-Tunnel ein Nadel\xc3\xb6hr dar: Sie verf\xc3\xbcgen derzeit nicht \xc3\xbcber das f\xc3\xbcr den unbegleiteten Kombinierten Verkehr (UKV) erforderliche Vier-Meter-Profil (P400). Ohne eine Anpassung dieser Tunnel auf das P400-Standardprofil kann kein durchg\xc3\xa4ngiger 4-Meter-Korridor auf der linken Rheinseite geschaffen werden. Die Erweiterung der Tunnelprofile in den Vogesen ist daher eine zwingende Voraussetzung, um die Strecke Antwerpen\xe2\x80\x93Basel vollumf\xc3\xa4nglich f\xc3\xbcr den G\xc3\xbcterverkehr nutzbar zu machen.\n\nAngesichts der strategischen Bedeutung dieses zweiten Zulaufs f\xc3\xbcr unsere Verlagerungspolitik soll die Schweiz aktiv Unterst\xc3\xbctzung leisten. Der Bundesrat wird deshalb beauftragt, Frankreich \xe2\x80\x93 analog zu den Unterst\xc3\xbctzungen beim 4-Meter-Korridor auf den Zulaufstrecken Luino und Simplon-Novara \xe2\x80\x93 finanzielle Mittel f\xc3\xbcr den Ausbau der Vogesen-Tunnel in Aussicht zu stellen. Diese Unterst\xc3\xbctzung soll durch die Umschichtung freiwerdender Gelder aus der Rollenden Landstrasse (RoLa) der Jahre 2026\xe2\x80\x932028 erfolgen. Mit dieser Investition sichert die Schweiz die rechtzeitige Realisierung eines belastbaren linksrheinischen NEAT-Zubringers. Dies st\xc3\xa4rkt die Ausweichkapazit\xc3\xa4ten im Transitverkehr, erh\xc3\xb6ht die Systemresilienz und tr\xc3\xa4gt entscheidend dazu bei, die Verlagerungsziele \xe2\x80\x93 insbesondere die Verlagerung des G\xc3\xbcterverkehrs von der Strasse auf die Schiene \xe2\x80\x93 nachhaltig zu erreichen.',
        'data': [(78, 1), (185, 1), (254, 1), (382, 1), (531, 1), (711, 2), (853, 1), (1090, 1), (1229, 1), (1354, 1), (1608, 0)],
    },
    {
        'text': b'Die Nichterf\xc3\xbcllung des Milit\xc3\xa4r- oder Zivildienstes kann verschiedene Ursachen haben, beispielsweise eine Dienstuntauglichkeit, die durch den milit\xc3\xa4r\xc3\xa4rztlichen Dienst festgestellt wurde, oder eine Dienstbefreiung aufgrund unentbehrlicher T\xc3\xa4tigkeit nach Artikel 18 des Milit\xc3\xa4rgesetzes (MG; SR 510.10).\n\n \n\nPersonen, welche nach Artikel 18 MG von der pers\xc3\xb6nlichen Milit\xc3\xa4r- oder Zivildienstleistung befreit sind, werden nach Artikel 4 Absatz 1 Buchstabe c des Bundesgesetzes \xc3\xbcber die Wehrpflichtersatzabgabe (WPEG; SR 661) auch von der Ersatzabgabe befreit. Voraussetzung f\xc3\xbcr diese Befreiung ist, dass sie diensttauglich sind und die Rekrutenschule erfolgreich abgeschlossen haben. Die Personen, die aufgrund unentbehrlicher T\xc3\xa4tigkeiten befreit werden, sind in Artikel 18 MG abschliessend aufgez\xc3\xa4hlt. Die Auflistung umfasst neben Angeh\xc3\xb6rigen der Polizei auch Angeh\xc3\xb6rige der Rettungsdienste, Feuerwehren, \xc3\xb6ffentlicher Transportunternehmungen, des Grenzwachtkorps, der Postdienste sowie Medizinalpersonen.\n\n \n\nHintergrund ist dabei die \xc3\x9cberlegung, dass ein Polizist, der vom pers\xc3\xb6nlichen Milit\xc3\xa4rdienst befreit ist, seine restliche \xc2\xabWehrpflicht\xc2\xbb zugunsten des \xc2\xabSicherheitssystems Schweiz\xc2\xbb durch seine T\xc3\xa4tigkeit als ziviler Polizist erf\xc3\xbcllt. Diese gesetzlich festgelegte Praxis basiert auf der Tatsache, dass das milit\xc3\xa4rische und zivile Sicherheitssystem auf dieselben Personen \xe2\x80\x93 hier ein Polizist \xe2\x80\x93 angewiesen ist. Ist hingegen ein Polizist nicht milit\xc3\xa4rdiensttauglich, kommt er lediglich f\xc3\xbcr die Funktion als Polizist und nicht auch als Angeh\xc3\xb6riger der Armee in Frage. Es braucht folglich keine Befreiung von der Milit\xc3\xa4r- oder Zivildienstleistung und somit auch nicht vom Wehrpflichtersatz, der anstelle dieser pers\xc3\xb6nlichen Dienstleistung erhoben wird.\n\n \n\nDas Bundesgericht urteilte im Fall 2A.300_2003 vom 24. Februar 2004, dass die unterschiedliche Behandlung von diensttauglichen und dienstuntauglichen Polizisten rechtens und nachvollziehbar sei. Die unterschiedliche Behandlung basiert auf einem pers\xc3\xb6nlichen Merkmal, das bei der Rekrutierung milit\xc3\xa4r\xc3\xa4rztlich gepr\xc3\xbcft wird und nicht auf einem Beruf, der freiwillig gew\xc3\xa4hlt wurde. W\xc3\xbcrden dienstuntaugliche Wehrpflichtige wegen ihrer beruflichen Stellung von der Ersatzpflicht befreit, so erg\xc3\xa4ben sich gewichtige Rechtsungleichheiten gegen\xc3\xbcber denjenigen Wehrpflichtigen, die aus pers\xc3\xb6nlichen Gr\xc3\xbcnden keinen Milit\xc3\xa4rdienst leisten und deshalb Milit\xc3\xa4rpflichtersatz leisten m\xc3\xbcssen (BGE 108 Ib 115 E. 5, S. 120). ',
        'data': [(305, 5), (562, 1), (687, 1), (808, 1), (1014, 5), (1257, 1), (1436, 1), (1594, 1), (1780, 5), (1979, 1), (2167, 1), (2502, 1)],
    },
    {
        'text': b'Im Juni 2021 verabschiedete das Parlament eine \xc3\x84nderung des Bundesgesetzes \xc3\xbcber die politischen Rechte (BPR; SR 161.1). Konkretisiert wurde diese \xc3\x84nderung im August 2022 durch die Verordnung \xc3\xbcber die Transparenz bei der Politikfinanzierung (VPofi, SR 161.18). Beide sind am 23. Oktober 2022 in Kraft getreten. Im 2024 veranlasste der Bundesrat die Evaluation der Umsetzung der neuen Transparenzregeln mit dem Ziel, m\xc3\xb6gliche M\xc3\xa4ngel aufzudecken und gegebenenfalls die erforderlichen Anpassungen vorzunehmen. Gest\xc3\xbctzt auf die Ergebnisse dieser Analyse wird der Bundesrat bei Bedarf dem Parlament konkrete Vorschl\xc3\xa4ge unterbreiten. Vor diesem Hintergrund w\xc3\xa4re es verfr\xc3\xbcht, eine Revision des BPR in Betracht zu ziehen, bevor die laufende Evaluation abgeschlossen ist und deren Ergebnisse eindeutig feststehen.',
        'data': [(121, 1), (263, 1), (313, 1), (511, 1), (634, 1), (813, 0)],
    },
    {
        'text': b'Der Bundesrat anerkennt das Anliegen der Motion und dessen Bedeutung. Das Engagement der Schweizer B\xc3\xbcrgerinnen und B\xc3\xbcrger f\xc3\xbcr ihr Land darf nicht dazu f\xc3\xbchren, dass sie ihre Arbeit verlieren. Allerdings sind sie jedoch f\xc3\xbcr diesen Fall durch das geltende Arbeitsrecht bereits gesch\xc3\xbctzt.\n\n \n\nSo verbietet es Artikel 336c Absatz 1 Buchstabe a OR dem Arbeitgeber, das Arbeitsverh\xc3\xa4ltnis nach Ablauf der Probezeit zu k\xc3\xbcndigen, w\xc3\xa4hrend der Arbeitnehmer obligatorischen Milit\xc3\xa4r- oder Schutzdienst oder Zivildienst leistet. Dieser Schutz erstreckt sich auf vier Wochen vor und nach der Dienstleistung, sofern diese mehr als elf Tage dauert. Gem\xc3\xa4ss Artikel 336c Absatz 2 OR ist eine w\xc3\xa4hrend dieser Sperrfrist erkl\xc3\xa4rte K\xc3\xbcndigung nichtig. Ist sie dagegen vor Beginn einer solchen Frist erfolgt, aber die K\xc3\xbcndigungsfrist bis dahin noch nicht abgelaufen, so wird deren Ablauf unterbrochen und erst nach Beendigung der Sperrfrist fortgesetzt.\n\n \n\nZudem ist die K\xc3\xbcndigung eines Arbeitsverh\xc3\xa4ltnisses gem\xc3\xa4ss Artikel 336 Absatz 1 Buchstabe e OR missbr\xc3\xa4uchlich, wenn eine Partei sie ausspricht, weil die andere Partei schweizerischen obligatorischen Milit\xc3\xa4r- oder Schutzdienst oder schweizerischen Zivildienst leistet oder eine nicht freiwillig \xc3\xbcbernommene gesetzliche Pflicht erf\xc3\xbcllt. Eine missbr\xc3\xa4uchliche K\xc3\xbcndigung verpflichtet den Arbeitgeber zur Ausrichtung einer Entsch\xc3\xa4digung von bis zu sechs Monatsl\xc3\xb6hnen, die vom Gericht festgesetzt wird (vgl. Art. 336a Abs. 1 und 2 OR).\n\n \n\nArtikel 336c OR sieht also einen umfassenden Schutz vor, der bereits mehrere Monate vor Beginn der Dienstleistung zum Tragen kommen kann, zum Beispiel wenn die K\xc3\xbcndigungsfrist zwei oder mehr Monate betr\xc3\xa4gt. Es muss dabei nicht bewiesen werden, dass die K\xc3\xbcndigung aufgrund der Dienstleistung ausgesprochen wurde. Ausserhalb der Sperrfrist nach Artikel 336c OR ist die oder der dienstpflichtige Arbeitnehmende durch Regelungen, die eine missbr\xc3\xa4uchliche K\xc3\xbcndigung verbieten, gesch\xc3\xbctzt. In diesem Fall muss die arbeitnehmende Person zwar beweisen, dass sie oder er aufgrund der Aus\xc3\xbcbung der Dienstpflicht entlassen wurde. Weil dies schwierig zu beweisen ist, hat die Rechtsprechung jedoch bereits den Indizienbeweis zugelassen. Entscheidende Indizien k\xc3\xb6nnen sich zum Beispiel aus dem zeitlichen Zusammenhang zwischen der Dienstanzeige und der K\xc3\xbcndigung ergeben.\n\n \n\nEs zeigt sich also, dass das Schweizer Recht bereits einen hohen und ausreichenden Schutz bietet. Der Bundesrat sieht keinen Bedarf, diesen noch weiter auszubauen, insbesondere auch mit Blick auf den bez\xc3\xbcglich anderer Gr\xc3\xbcnde und Situationen bestehenden Schutz.',
        'data':[(69, 1), (194, 1), (290, 5), (523, 1), (640, 1), (739, 1), (941, 5), (1286, 1), (1484, 5), (1697, 1), (1803, 1), (1977, 1), (2113, 1), (2219, 1), (2355, 5), (2457, 1), (2622, 0)],
    },
    {
        'text': b'\n\nDie Stellungnahme ist identisch mit der Stellungnahme zur gleichlautenden Motion 24.4323 Fraktion V \xc2\xab\xc3\x84nderungen der Internationalen Gesundheitsvorschriften: Den demokratischen Prozess gew\xc3\xa4hrleisten\xc2\xbb.\n\n \n\nDie juristische Analyse bez\xc3\xbcglich der Vertragsabschlusskompetenz der Anpassungen der Internationale Gesundheitsvorschriften (2005) (SR 0.818.103; nachfolgend: IGV) wird zurzeit durch die zust\xc3\xa4ndigen Stellen der Bundesverwaltung vorgenommen.\n\n \n\nDer Bundesrat hat am 13. November 2024 eine Vernehmlassung zu den Anpassungen der IGV er\xc3\xb6ffnet (www.fedlex.admin.ch > Vernehmlassungen > Laufende Vernehmlassungen > Vernehmlassung 2024/87), was auch zur umfassenden Information der \xc3\x96ffentlichkeit \xc3\xbcber die verabschiedeten Anpassungen und zur Konsultation der Kantone beitr\xc3\xa4gt. Die Vernehmlassung l\xc3\xa4uft bis am 27. Februar 2025. Ein vorsorgliches \xc2\xabOpting out\xc2\xbb vor der Auswertung der laufenden Vernehmlassung und vor Abschluss der juristischen Analyse zur Vertragsabschlusskompetenz ist aus Sicht des Bundesrates nicht notwendig. Nach Kenntnisnahme des Ergebnisberichts aus der Vernehmlassung wird der Bundesrat \xc3\xbcber die n\xc3\xa4chsten Schritte unter Ber\xc3\xbccksichtigung der Fristen entscheiden.\n\n \n\nDie Motion verlangt, dass dem Parlament unabh\xc3\xa4ngig vom Ergebnis der Analyse der Vertragsabschlusskompetenz die Anpassungen der IGV zur Genehmigung unterbreitet und dem fakultativen Referendum unterstellt werden. Damit setzt sich die Motion \xc3\xbcber die geltenden Regelungen und Verfahren bez\xc3\xbcglich der Kompetenz zum Abschluss internationaler Abkommen (siehe Art. 166 Abs. 2 und 184 Abs. 1 und 2 BV, Art. 24 Parlamentsgesetz [ParlG; SR 171.10] und Art. 7a Regierungs- und Verwaltungsorganisationsgesetz [RVOG; SR 172.010]) hinweg.\n\n\n\nDer Bundesrat beantragt die Ablehnung der Motion.',
        'data': [(0, 2), (205, 5), (452, 5), (786, 1), (837, 1), (1039, 1), (1199, 5), (1416, 1), (1732, 4), (1785, 0)],
    },
    {
        'text': b'Die Motion verlangt, dass eine digitale Plattform f\xc3\xbcr den Schriftverkehr zwischen den Lobbyistinnen und Lobbyisten und den Ratsmitgliedern geschaffen wird. Ziel ist es, den Schriftverkehr zu zentralisieren, damit die Mailboxen der Ratsmitglieder nicht mit E-Mails von Lobbyistinnen und Lobbyisten \xc3\xbcberquellen, und gleichzeitig die Transparenz zu f\xc3\xb6rdern, indem beispielsweise auch akkreditierten Medienschaffenden Zugang zur Plattform gew\xc3\xa4hrt wird.\n\n \n\nDiese Motion wirft zahlreiche grunds\xc3\xa4tzliche und praktische Fragen auf.\n\n \n\nEine auf Freiwilligkeit beruhende Plattform ist zwar denkbar, doch ist es unwahrscheinlich, dass alle Lobbyistinnen und Lobbyisten bereit w\xc3\xa4ren, ihre Unterlagen, Einladungen und Positionspapiere dort abzulegen. Es best\xc3\xbcnde somit die Gefahr, dass eine solche Plattform unvollst\xc3\xa4ndig ist und bisweilen zu Doppelspurigkeiten f\xc3\xbchrt. Auf jeden Fall aber w\xc3\xbcrde sie die Arbeit der Ratsmitglieder zus\xc3\xa4tzlich erschweren und damit genau das Gegenteil von dem bewirken, was mit der Motion beabsichtigt wird.\n\n \n\nAlternativ k\xc3\xb6nnte laut der Motion eine Rechtsgrundlage geschaffen werden, welche die Lobbyistinnen und Lobbyisten verpflichtet, ihren Schriftverkehr ausschliesslich auf dieser Plattform abzulegen, und den Versand von E-Mails und Briefpost an die Ratsmitglieder untersagt. Eine solche Massnahme w\xc3\xa4re aus Sicht der Meinungs- und Vereinigungsfreiheit sowie des Schutzes der Privatsph\xc3\xa4re problematisch. Beim gegenw\xc3\xa4rtigen Stand der Dinge ist nicht ersichtlich, kraft welcher Befugnis das Parlament eine solche Verpflichtung einf\xc3\xbchren k\xc3\xb6nnte. Zudem k\xc3\xb6nnte es deren Einhaltung nur durch die Schaffung eines aufw\xc3\xa4ndigen Kontrollinstruments gew\xc3\xa4hrleisten. Dar\xc3\xbcber hinaus m\xc3\xbcsste in der Rechtsgrundlage der Begriff \xc2\xabLobby\xc2\xbb genau definiert werden. W\xc3\xbcrden auch Dachverb\xc3\xa4nde, Beratungsgesellschaften, Gewerkschaften, Berufsverb\xc3\xa4nde, NGO, parlamentarische Gruppen (Art. 63 ParlG), informelle Kreise sowie Interessenvertreterinnen und -vertreter der Kantone (siehe pa. Iv. 23.425 Masshardt [\xc2\xabTransparentes Lobbying der Kantone\xc2\xbb]) unter diesen Begriff fallen? Hinzu kommt, dass bestimmte Ratsmitglieder selbst als Vertreterinnen oder Vertreter von Interessengruppen t\xc3\xa4tig sind. Ferner ist darauf hinzuweisen, dass es nicht Aufgabe des Staates ist, den Lobbyistinnen und Lobbyisten bei ihren Beziehungen zum Parlament die Arbeit zu erleichtern.',
        'data': [(156, 1), (452, 5), (529, 5), (745, 1), (866, 1), (1036, 5), (1313, 1), (1442, 1), (1584, 1), (1697, 1), (1790, 1), (2102, 1), (2221, 1), (2387, 0)],
    },
]

FILLERS = [' ', '\n', '\n\n', '  ', '\t']


class SentenceSplitterDataset(Dataset):

    def __init__(self, train=True, transform=None, min_length=512, max_length=2048, output_size=256, disable_pytorch=False):
        super(SentenceSplitterDataset, self).__init__()
        # train is ignored for now
        # check if arguments are valid
        if min_length > max_length:
            raise ValueError("min_length must be less than or equal than max_length")
        self._train = train
        self._transform = transform
        self._min_length = min_length
        self._max_length = max_length
        self._output_size = output_size
        self._disable_pytorch = disable_pytorch
        self._string = b''
        self._void: t.List[int] = []
        self._ends: t.List[int] = []
        for set_index, data_set in enumerate(TRAINING_DATA):
            string_length = len(self._string)
            if len(self._string) > 0:
                filler = FILLERS[(((set_index + 1) ** 2) - 1) % len(FILLERS)].encode('utf-8')
                self._string += filler
                for i in range(len(filler)):
                    self._void.append(string_length + i)
            string_length = len(self._string)
            self._string += data_set['text']
            for point in data_set['data']:
                if point[0] > 0:
                    self._ends.append(string_length + point[0])
                if point[1] > 0:
                    for i in range(point[1]):
                        self._void.append(string_length + point[0] + i)

    def debug(self):
        print(f"Min length: {self._min_length}, Max length: {self._max_length}, String length: {len(self._string)}, Void length: {len(self._void)}, End length: {len(self._ends)}, Void: {self._void}, Ends: {self._ends}")

    def _cumulative_count(self, l):  # written by `DeepSeek-R1-0528`
        n = len(self._string)
        l_high_bound = min(n, self._max_length)
        if self._min_length > l_high_bound:
            return 0
        if l < self._min_length:
            return 0
        l_val = min(l, l_high_bound)
        terms = l_val - self._min_length + 1
        return terms * (2 * n - self._min_length - l_val + 2) // 2

    def __len__(self):
        n = len(self._string)
        l_bound = min(n, self._max_length)
        return self._cumulative_count(l_bound)

    def __getitem__(self, index):  # partially written by `DeepSeek-R1-0528`
        total = len(self)
        if index < 0 or index >= total:
            raise IndexError("index out of range")

        n = len(self._string)
        low = self._min_length
        high = min(n, self._max_length)
        l_found = high + 1

        while low <= high:
            mid = (low + high) // 2
            f_mid = self._cumulative_count(mid)
            if f_mid <= index:
                low = mid + 1
            else:
                l_found = mid
                high = mid - 1

        f_prev = self._cumulative_count(l_found - 1)
        start_index = index - f_prev
        data = []
        for i in range(start_index, start_index + l_found):
            if i in self._ends:
                end = i - start_index
                voids = 0
                for j in range(i, start_index + l_found):
                    if j in self._void and end + voids < start_index + l_found:
                        voids += 1
                    else:
                        break
                data.append([end, voids])
        text = self._string[start_index:start_index + l_found]
        text_length = len(text)
        if not self._disable_pytorch:
            text += b'\x00' * (INPUT_SIZE - text_length)
        if data[-1][0] + data[-1][1] >= text_length:
            data[-1][1] = INPUT_SIZE - data[-1][0]
        output = []
        for point in data:
            output.append(point[0])
            output.append(point[1])
        if self._disable_pytorch:
            return text, output
        while len(output) < self._output_size:
            output.append(0)
        f = tensor([float(number) for number in text], device='cuda'), tensor([float(number) for number in output], device='cuda')
        if self._transform:
            f = self._transform(f[0]), self._transform(f[1])
        return f


def main() -> None:
    for _i, data_set in enumerate(TRAINING_DATA):
        print(f"Dataset {_i}")
        text = data_set['text']
        data = data_set['data']
        for i, _ in enumerate(data):
            if data[i][0] != 0:
                start = 0 if i == 0 else (data[i - 1][0] + data[i - 1][1])
                end = data[i][0]
                print(f"[{start}:{end}] " + text[start:end].decode('utf-8'))
            else:
                end = data[i][0]
            start = end
            end = end + data[i][1]
            print(f"[{start}:{end}] " + text[start:end].decode('utf-8'))


if __name__ == '__main__':
    main()
